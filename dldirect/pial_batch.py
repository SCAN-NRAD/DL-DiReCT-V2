#!/usr/bin/env python
"""Run pial_pipeline.reconstruct over many preps in ONE process.

WHY THIS EXISTS

Each `python -m dldirect.pial_pipeline` invocation pays three fixed costs that
have nothing to do with the subject:

  interpreter + torch import      3.79 s
  CUDA context creation           0.18 s
  torch.compile / inductor warm   7.2  s   (see below)

Measured by running the SAME prep three times in one process, so the work is
identical and the only variable is warmth (RTX 6000 Ada, padded OAS30001 prep,
surface-pv segmentation, the pipeline's default topology):

    run 1   55.80 s
    run 2   48.67 s
    run 3   48.63 s

The 7.2 s is paid once and never again -- run 2 and run 3 agree to 0.04 s. So a
batch of N subjects in one process saves roughly (3.79 + 0.18 + 7.2) * (N - 1)
seconds against N separate invocations, i.e. ~11 s per subject after the first,
against a ~49 s steady-state subject. That is a ~19% saving on a long batch.

THE WARM-UP CANNOT BE MOVED OFF SUBJECT 1. A `warm_up()` that called the four
compiled kernels (gaussian_smooth_3d, gaussian_gradient_3d, warp_image,
compose_fields) on a 32^3 dummy volume was tried and REMOVED: it cost 16.4 s by
itself and left subject 1 no faster (62.7 s with it against 62.1 s without), for
a net loss of 3.7 s over a two-subject batch. Compiling at a toy shape does not
specialise the graph the real shapes then use, despite dynamic=True. Do not
re-add it; pay the warm-up on the first real subject.

That the penalty belongs to the POSITION and not to a subject was checked by
running the same two preps in both orders -- first slot 62.1 s / 64.1 s, second
slot 58.1 s / 54.0 s. Whichever prep goes first pays it. (This is the trap in
[[benchmark-gpu-solve-alternating]]: a single A-then-B comparison charges the
whole warm-up to A.)

WHAT IS NOT SHARED BETWEEN SUBJECTS

Nothing derived from the data. Each subject re-reads its own prep and rebuilds
its own segmentation, white surfaces and velocity field; `reconstruct` holds no
state across calls. The only things reused are the compiled kernels, the CUDA
context and the caching allocator's blocks -- none of which carry subject data.
The allocator is deliberately NOT emptied between subjects (that is what keeps
allocation warm); pass --empty-cache if a subject's peak memory would otherwise
not fit.

STEADY STATE IS DOMINATED BY THE SEGMENTATION, NOT THE SOLVE: of the 48.6 s,
the surface-pv segmentation and white-surface build take 39.0 s and the solve
plus both propagations take ~9.6 s. Batching does not touch that 39 s.
"""

import argparse
import glob
import os
import sys
import time
import traceback

import torch

from . import pial_pipeline as pp
from . import pial_clean as pc
from . import solve_grid
from . import wm_surface


def _out_dir_for(prep_dir, out_name, out_root):
    """Where this subject's surfaces go.

    Default is `<prep>/<out_name>`, which keeps each subject's outputs beside
    the prep they came from. With --out-root the subject's directory name is
    used instead, so a batch writes one flat tree that is easy to iterate over.
    """
    if out_root is None:
        return os.path.join(prep_dir, out_name)
    return os.path.join(out_root, os.path.basename(os.path.normpath(prep_dir)),
                        out_name)


def run_batch(prep_dirs, out_name='field_pial', out_root=None, skip_existing=False,
              hemis=('lh', 'rh'), empty_cache=False, verbose=True,
              subject_verbose=None, report=True, on_error='continue',
              **reconstruct_kwargs):
    """reconstruct() over `prep_dirs` in this process. Returns a result list.

    Each entry is a dict with `prep_dir`, `out_dir`, `seconds`, and either
    `status='ok'` or `status='failed'` plus `error`. The surfaces themselves are
    NOT kept: a 36-hemisphere batch would hold ~1.5 GB of vertices for no
    reason, and they have already been written to `out_dir`.

    verbose          the batch's own progress and the closing summary. The
                     summary is the point of a batch run, so --quiet turns
                     `subject_verbose` off and leaves this on.
    subject_verbose  what each reconstruct() prints (segmentation progress,
                     solver iterations). Defaults to `verbose`.
    on_error         'continue' (default) records the failure and moves to the
                     next subject; 'stop' re-raises. A batch left running
                     unattended wants the first, a debugging run the second.
    """
    if subject_verbose is None:
        subject_verbose = verbose
    prep_dirs = list(prep_dirs)
    results = []
    t_batch = time.time()

    for i, prep in enumerate(prep_dirs, 1):
        out_dir = _out_dir_for(prep, out_name, out_root)
        want = ['%s.pial' % h for h in hemis]
        if reconstruct_kwargs.get('stats'):
            # otherwise --skip-existing on a batch rerun that ADDS --stats skips
            # every subject and writes no CSV at all
            want.append('result-thick-field.csv')
        done = all(os.path.exists(os.path.join(out_dir, f)) for f in want)
        if skip_existing and done:
            if verbose:
                print('[%d/%d] %s: skipped (surfaces present)'
                      % (i, len(prep_dirs), os.path.basename(os.path.normpath(prep))))
            results.append(dict(prep_dir=prep, out_dir=out_dir, seconds=0.0,
                                status='skipped'))
            continue

        if verbose:
            print('[%d/%d] %s' % (i, len(prep_dirs),
                                  os.path.basename(os.path.normpath(prep))))
        t = time.time()
        try:
            pp.reconstruct(prep, hemis=tuple(hemis), out_dir=out_dir,
                           verbose=subject_verbose, report=report,
                           **reconstruct_kwargs)
        except Exception as exc:
            dt = time.time() - t
            if on_error == 'stop':
                raise
            print('  FAILED after %.1fs: %s: %s' % (dt, type(exc).__name__, exc),
                  file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            results.append(dict(prep_dir=prep, out_dir=out_dir, seconds=dt,
                                status='failed', error='%s: %s'
                                % (type(exc).__name__, exc)))
        else:
            dt = time.time() - t
            if verbose:
                print('  %.1fs -> %s' % (dt, out_dir))
            results.append(dict(prep_dir=prep, out_dir=out_dir, seconds=dt,
                                status='ok'))
        # Between subjects, not inside one: releasing the allocator's blocks is
        # what makes the NEXT allocation cold, so it is off unless asked for.
        if empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    total = time.time() - t_batch
    ok = [r for r in results if r['status'] == 'ok']
    if verbose:
        print()
        print('%d/%d ok, %d failed, %d skipped in %.1fs'
              % (len(ok), len(prep_dirs),
                 sum(r['status'] == 'failed' for r in results),
                 sum(r['status'] == 'skipped' for r in results), total))
        if ok:
            t_each = [r['seconds'] for r in ok]
            rest = t_each[1:]
            print('per subject: first %.1fs, min %.1fs, max %.1fs'
                  % (t_each[0], min(t_each), max(t_each)))
            if rest:
                # first - median is the warm-up PLUS whatever else made subject 1
                # differ, so it is reported as a difference and not attributed.
                med = sorted(rest)[len(rest) // 2]
                print('after the first: median %.1fs (first - median = %+.1fs; the '
                      'warm-up is ~6-8s of that, the rest is subject and load)'
                      % (med, t_each[0] - med))
    return results


def collect_preps(patterns, list_file=None):
    """Prep directories from globs and/or a file of one path per line.

    A prep is recognised by the files `reconstruct` actually reads, so a parent
    directory full of subjects can be globbed without hand-listing them and a
    non-prep directory is reported rather than failing 40 s into the solve.
    """
    paths = []
    if list_file:
        with open(list_file) as fh:
            paths += [ln.strip() for ln in fh if ln.strip()
                      and not ln.lstrip().startswith('#')]
    for pat in patterns or ():
        hits = sorted(glob.glob(pat))
        paths += hits if hits else [pat]

    good, bad = [], []
    for p in paths:
        need = [os.path.join(p, 'seg_Left-Cerebral-Cortex.nii.gz'),
                os.path.join(p, 'label_def.csv')]
        (good if all(os.path.exists(f) for f in need) else bad).append(p)
    for p in bad:
        print('not a prep (no seg_Left-Cerebral-Cortex.nii.gz / label_def.csv): %s'
              % p, file=sys.stderr)
    # dedupe, keep order
    seen, out = set(), []
    for p in good:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('prep_dirs', nargs='*',
                   help='prep directories, or globs matching them')
    p.add_argument('--list', dest='list_file',
                   help='file with one prep directory per line (# comments allowed)')
    p.add_argument('--out-name', default='field_pial',
                   help='subdirectory written inside each prep (default field_pial)')
    p.add_argument('--out-root',
                   help='write <root>/<subject>/<out-name> instead of inside the prep')
    p.add_argument('--skip-existing', action='store_true',
                   help='skip subjects whose ?h.pial already exist (resume a batch)')
    p.add_argument('--empty-cache', action='store_true',
                   help='release cached GPU blocks between subjects (slower; use '
                        'only if a subject would not otherwise fit)')
    p.add_argument('--stop-on-error', action='store_true',
                   help='re-raise instead of recording the failure and continuing')
    p.add_argument('--quiet', action='store_true',
                   help='one line per subject: suppress the solver and '
                        'segmentation chatter, keep the summary')
    p.add_argument('--csv', help='write per-subject status and timing here')
    # --- the reconstruct knobs, same names and defaults as pial_pipeline ---
    p.add_argument('--hemi', nargs='+', default=['lh', 'rh'], choices=['lh', 'rh'])
    p.add_argument('--velocity-sigma', type=float, default=pc.VELOCITY_SIGMA,
                   help='ANTs -b, the velocity smoothing sigma (default %.2f)'
                        % pc.VELOCITY_SIGMA)
    p.add_argument('--segmentation', default='logits',
                   choices=['logits', 'surface-pv'])
    # No default here ON PURPOSE. A default set in this file SHADOWS
    # reconstruct's, because it is passed explicitly -- which is exactly how
    # pial_pipeline's --topology came to be inert, and how a 60-subject
    # rebuild meant to test the nighres default silently ran on gpu instead.
    # None means 'do not pass it', so the pipeline's own default governs.
    p.add_argument('--wm-from-surface', action='store_true',
                   help="'logits' only: reconcile the WM label against the "
                        'white surface before solving (off by default)')
    p.add_argument('--topology', default=None, choices=['nighres', 'gpu', 'none'])
    p.add_argument('--nsmooth', type=int, default=wm_surface.NSMOOTH_DEFAULT)
    p.add_argument('--solve-margin', type=int, default=solve_grid.MARGIN,
                   help='voxels of background guaranteed around the cerebrum for the '
                        'solve and the propagation (default %d; -1 keeps the grid as '
                        'given)' % solve_grid.MARGIN)
    p.add_argument('--propagate-on', default='cuda', choices=['cpu', 'cuda'])
    p.add_argument('--thickness', action='store_true',
                   help='also compute the DiReCT thickness map')
    # default=None so this cannot SHADOW reconstruct's own default when passed
    # explicitly -- the mistake --topology made, two lines up.
    p.add_argument('--no-ribbon-correction', dest='correct_ribbon',
                   action='store_false', default=None)
    p.add_argument('--reuse-white', default=None,
                   help="directory name inside each prep holding ?h.white from an "
                        "earlier run to start from (e.g. field_pial_sigma0.65), "
                        "instead of rebuilding the white surfaces")
    p.add_argument('--stats', action='store_true', default=None,
                   help='also write regional_stats\' result-thick-<metric>.csv beside '
                        'each subject\'s surfaces, off the solve already in memory')
    p.add_argument('--float64', action='store_true')
    args = p.parse_args()

    preps = collect_preps(args.prep_dirs, args.list_file)
    if not preps:
        p.error('no prep directories found')
    print('%d prep%s to process' % (len(preps), '' if len(preps) == 1 else 's'))

    res = run_batch(
        preps, out_name=args.out_name, out_root=args.out_root,
        skip_existing=args.skip_existing, hemis=tuple(args.hemi),
        empty_cache=args.empty_cache,
        verbose=True, subject_verbose=not args.quiet, report=not args.quiet,
        on_error='stop' if args.stop_on_error else 'continue',
        propagate_on=args.propagate_on,
        dtype=torch.float64 if args.float64 else torch.float32,
        compute_thickness=args.thickness, nsmooth=args.nsmooth,
        segmentation=args.segmentation,
        **({} if args.topology is None else {'topology': args.topology}),
        **({} if args.stats is None else {'stats': args.stats}),
        **({} if args.reuse_white is None else {'reuse_white': args.reuse_white}),
        **({} if args.correct_ribbon is None else {'correct_ribbon': args.correct_ribbon}),
        velocity_sigma=args.velocity_sigma,
        wm_from_surface=args.wm_from_surface,
        solve_margin=None if args.solve_margin < 0 else args.solve_margin)

    if args.csv:
        import csv as _csv
        with open(args.csv, 'w', newline='') as fh:
            w = _csv.DictWriter(fh, fieldnames=['prep_dir', 'out_dir', 'status',
                                                'seconds', 'error'])
            w.writeheader()
            for r in res:
                w.writerow({k: r.get(k, '') for k in w.fieldnames})
        print('wrote %s' % args.csv)

    return 1 if any(r['status'] == 'failed' for r in res) else 0


if __name__ == '__main__':
    sys.exit(main())
