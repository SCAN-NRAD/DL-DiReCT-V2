"""Bayesian optimization over field_pial_prototype's PipelineConfig.

STATUS: research prototype, not wired into the CLI or default pipeline —
same status as field_pial_prototype.py itself, which this module depends
on and does not modify.

Treats one full solve+propagate+evaluate run (config_loss(), see
field_pial_prototype.py) as an expensive black-box function of a handful
of continuous parameters, and searches it with a standard Gaussian-process
/ expected-improvement loop built from scikit-learn (already a dependency
of this environment — no new package was added for this). This is a
from-scratch ~80-line GP-EI implementation, not a wrapper around a
dedicated library (scikit-optimize / optuna / bayes_opt are not installed
here); if a real search budget is available, switching to one of those is
a reasonable upgrade — the objective function below is written to be
library-agnostic.

Each evaluation costs a real GPU solve (order of a minute) plus a CPU
propagate/evaluate pass per hemisphere (order of a minute each), so the
search space is deliberately small and the default budget is small too —
this is meant to demonstrate the harness and give a short list of
promising configurations to inspect by hand, not to run unattended for
hours. Widen SEARCH_SPACE or raise --n-iter for a more serious search.

Usage:
    python -m dldirect.field_pial_optimize \\
        --prep-dir <...> --surf-dir <.../surf> --out-dir <...> \\
        --hemi lh --n-init 4 --n-iter 8

Writes a CSV of every evaluated configuration and its loss/metrics to
<out-dir>/search_log.csv, and prints the best configuration found.
"""
import argparse
import csv
import dataclasses
import os

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from .field_pial_prototype import (
    PipelineConfig, BEST_CONFIG, run_pipeline, config_loss,
    load_gm_wm_probability, build_seg_maps, make_transforms,
)
import nibabel as nib


# (name, low, high) — continuous parameters only. The boolean toggles
# (use_normal_gate / use_sulcal_sheet / use_constrained_start) are held ON
# at BEST_CONFIG's settings: turning a fix off entirely is a much larger,
# non-smooth jump that a GP surrogate is poorly suited to model alongside
# continuous parameters, and it's a cheap grid/ablation question on its
# own (see field_pial_prototype.py's naive/best comparison for the
# all-off vs all-on endpoints already measured).
SEARCH_SPACE = [
    ('smoothing_sigma', 0.20, 0.80),
    ('gate_sigma_scale', 1.0, 3.0),
    ('gate_coherence_threshold', 0.5, 0.9),
    ('dip_threshold', 0.70, 1.00),
    ('constrained_floor', -1.0, -0.2),
]


def config_from_x(x, base=BEST_CONFIG):
    """Map a normalized [0,1]^d vector to a PipelineConfig, holding
    everything not in SEARCH_SPACE at `base`'s value."""
    kwargs = {}
    for (name, lo, hi), xi in zip(SEARCH_SPACE, x):
        kwargs[name] = float(lo + xi * (hi - lo))
    return dataclasses.replace(base, **kwargs)


def x_bounds():
    return np.array([[0.0, 1.0]] * len(SEARCH_SPACE))


class Objective:
    """Wraps one (seg, gm/wm, hemisphere) problem as f(x) -> loss, caching
    nothing between calls — every call is a real GPU solve."""

    def __init__(self, prep_dir, surf_dir, hemi, out_dir, tag_prefix):
        case = build_case(prep_dir, surf_dir, hemi)
        (self.gm_prob, self.wm_prob, self.ref_img, self.seg, self.gmT, self.wmT,
         self.tovox, self.totkr, self.hemi_surfaces) = case
        self.out_dir = out_dir
        self.tag_prefix = tag_prefix
        self.n_calls = 0

    def __call__(self, x):
        config = config_from_x(x)
        tag = '%s%03d' % (self.tag_prefix, self.n_calls)
        self.n_calls += 1
        results = run_pipeline(
            config, self.seg, self.gmT, self.wmT, self.gm_prob, self.wm_prob, self.ref_img,
            self.tovox, self.totkr, self.hemi_surfaces,
            os.path.join(self.out_dir, tag + '_'), out_dir=None, tag=tag)
        losses, all_metrics = [], {}
        for hemi, metrics in results.items():
            n_verts = len(self.hemi_surfaces[hemi][0])
            losses.append(config_loss(metrics, n_verts))
            all_metrics[hemi] = metrics
        return float(np.mean(losses)), config, all_metrics


def build_case(prep_dir, surf_dir, hemi):
    """Load and precompute everything about one scan that doesn't depend
    on the PipelineConfig being searched — the expensive part (the GPU
    solve) still happens fresh in run_pipeline() on every call."""
    gm_prob, wm_prob, ref_img = load_gm_wm_probability(prep_dir)
    seg, gmT, wmT = build_seg_maps(gm_prob, wm_prob)
    tovox, totkr = make_transforms(ref_img)
    hemi_surfaces = {
        h: nib.freesurfer.io.read_geometry(os.path.join(surf_dir, '%s.white' % h)) for h in hemi
    }
    return gm_prob, wm_prob, ref_img, seg, gmT, wmT, tovox, totkr, hemi_surfaces


def load_cases_file(path):
    """Each non-empty, non-'#' line: '<prep_dir>\\t<surf_dir>' (or any
    whitespace-separated pair) — one scan per line."""
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            prep_dir, surf_dir = line.split()[:2]
            cases.append((prep_dir, surf_dir))
    return cases


class MultiCaseObjective:
    """Like Objective, but averages the loss across several independently
    preprocessed scans (subjects, acquisitions, ...) for the SAME
    PipelineConfig — one real GPU solve per case, per call, so the cost of
    a single evaluation scales linearly with the number of cases. This is
    what makes a search result about the CONFIGURATION rather than about
    one scan's particular quirks."""

    def __init__(self, cases, hemi, out_dir, tag_prefix, verbose=True):
        self.cases = []
        for i, (prep_dir, surf_dir) in enumerate(cases):
            if verbose:
                print("loading case %d/%d: %s" % (i + 1, len(cases), prep_dir), flush=True)
            self.cases.append((os.path.basename(os.path.normpath(prep_dir)), build_case(prep_dir, surf_dir, hemi)))
        self.out_dir = out_dir
        self.tag_prefix = tag_prefix
        self.n_calls = 0
        self.verbose = verbose

    def __call__(self, x):
        config = config_from_x(x)
        call_tag = '%s%03d' % (self.tag_prefix, self.n_calls)
        self.n_calls += 1
        per_case_losses, all_metrics = [], {}
        for case_name, (gm_prob, wm_prob, ref_img, seg, gmT, wmT, tovox, totkr, hemi_surfaces) in self.cases:
            results = run_pipeline(
                config, seg, gmT, wmT, gm_prob, wm_prob, ref_img, tovox, totkr, hemi_surfaces,
                os.path.join(self.out_dir, '%s_%s_' % (call_tag, case_name)), out_dir=None, tag=call_tag)
            case_losses = []
            for hemi, metrics in results.items():
                n_verts = len(hemi_surfaces[hemi][0])
                case_losses.append(config_loss(metrics, n_verts))
                all_metrics['%s.%s' % (case_name, hemi)] = metrics
            case_loss = float(np.mean(case_losses))
            per_case_losses.append(case_loss)
            if self.verbose:
                print("    %s: loss=%.4f" % (case_name, case_loss), flush=True)
        return float(np.mean(per_case_losses)), config, all_metrics


def expected_improvement(x, gp, y_best, xi=0.01):
    mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)
    sigma = max(float(sigma[0]), 1e-9)
    improvement = y_best - float(mu[0]) - xi
    z = improvement / sigma
    return improvement * norm.cdf(z) + sigma * norm.pdf(z)


def propose_next(gp, y_best, bounds, n_candidates=4000, rng=None):
    rng = rng or np.random.default_rng()
    candidates = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_candidates, len(bounds)))
    ei = np.array([expected_improvement(c, gp, y_best) for c in candidates])
    return candidates[np.argmax(ei)]


def optimize(objective, n_init=4, n_iter=8, seed=0, log_path=None):
    """Minimal GP/expected-improvement Bayesian optimization loop. Returns
    the list of (loss, config, metrics) tried, best first."""
    rng = np.random.default_rng(seed)
    bounds = x_bounds()
    xs, ys, records = [], [], []

    log_f = None
    if log_path:
        log_f = open(log_path, 'w', newline='')
        writer = csv.writer(log_f)
        writer.writerow(['iter', 'loss'] + [s[0] for s in SEARCH_SPACE])

    def evaluate(x):
        loss, config, metrics = objective(x)
        xs.append(x)
        ys.append(loss)
        records.append((loss, config, metrics))
        print("[%2d] loss=%.4f  %s" % (
            len(xs) - 1, loss,
            {s[0]: round(getattr(config, s[0]), 4) for s in SEARCH_SPACE}), flush=True)
        if log_f:
            writer.writerow([len(xs) - 1, loss] + [getattr(config, s[0]) for s in SEARCH_SPACE])
            log_f.flush()
        return loss

    for x in rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_init, len(bounds))):
        evaluate(x)

    kernel = Matern(length_scale=np.ones(len(bounds)) * 0.3, nu=2.5) + WhiteKernel(noise_level=1e-3)
    for i in range(n_iter):
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed)
        gp.fit(np.array(xs), np.array(ys))
        x_next = propose_next(gp, min(ys), bounds, rng=rng)
        evaluate(x_next)

    if log_f:
        log_f.close()
    records.sort(key=lambda r: r[0])
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--prep-dir', help='single-case mode: one scan\'s prep dir')
    p.add_argument('--surf-dir', help='single-case mode: that scan\'s surf/ dir')
    p.add_argument('--cases-file',
                    help='multi-case mode: a file with one "<prep_dir> <surf_dir>" pair per line — '
                         'loss is averaged across all of them per configuration, so a result is '
                         'about the configuration rather than one scan\'s particular quirks. '
                         'Overrides --prep-dir/--surf-dir if given.')
    p.add_argument('--hemi', nargs='+', default=['lh'], choices=['lh', 'rh'],
                    help='hemisphere(s) to evaluate per case (default: lh only, for speed — '
                         'each extra hemisphere roughly doubles the per-case cost, and every '
                         'case is evaluated on every iteration)')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--n-init', type=int, default=4, help='random configurations before GP fitting starts')
    p.add_argument('--n-iter', type=int, default=8, help='GP/expected-improvement proposals after that')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.cases_file:
        cases = load_cases_file(args.cases_file)
        print("multi-case search: %d case(s) from %s, %s per iteration" %
              (len(cases), args.cases_file, '+'.join(args.hemi)), flush=True)
        objective = MultiCaseObjective(cases, args.hemi, args.out_dir, tag_prefix='search_')
    else:
        if not (args.prep_dir and args.surf_dir):
            p.error('either --cases-file, or both --prep-dir and --surf-dir, are required')
        objective = Objective(args.prep_dir, args.surf_dir, args.hemi, args.out_dir, tag_prefix='search_')
    records = optimize(objective, n_init=args.n_init, n_iter=args.n_iter, seed=args.seed,
                        log_path=os.path.join(args.out_dir, 'search_log.csv'))

    print("\n=== best of %d evaluations ===" % len(records))
    best_loss, best_config, best_metrics = records[0]
    print("loss=%.4f" % best_loss)
    print(best_config)
    print(best_metrics)


if __name__ == '__main__':
    main()
