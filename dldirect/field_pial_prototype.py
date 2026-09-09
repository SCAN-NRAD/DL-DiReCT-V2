"""Prototype: pial surface reconstruction by forward-integrating the DiReCT
velocity field, instead of the shipped FreeSurfer / nighres pial pipeline.

STATUS: research prototype, not wired into the CLI or the default pipeline.
It is not imported by any other module in this package. Run it directly
(see `main()` / `--help`) against an existing DL+DiReCT-V2 output directory
to reproduce the investigation's best configuration and compare it against
a "naive" baseline (shipped solver defaults, single-pass propagation, no
gate, no CSF sheet).

Background
----------
Propagating the white surface along DiReCT's plain velocity field gives a
reasonable pial surface on average, but a persistent minority of vertices
(a few percent) either transit through white matter, cross a sulcal gap
into the opposing gyral bank, or sit in a region where the field is
identically zero and never move at all. Investigating this on a single
test subject (bert, left hemisphere) found three independent, additive
fixes, implemented below:

1. `install_normal_gate()` — the shipped velocity smoothing
   (`velocity_smooth_gate='direction'`) decides which voxels to protect
   from cross-blade averaging by looking at the *field's own* direction —
   which is exactly what's unreliable in a thin gyral blade. Gating
   instead on the interface normal (the gradient of the WM signed-distance
   transform, a purely geometric quantity) removes that circularity.
   Combined with a flatter, soft-weighted neighbour kernel (same physical
   radius, less centre-dominance).

   CORRECTION: an earlier version of this note claimed the gate "reduces
   folding". Ablated on the development subject (--no-normal-gate, everything
   else default), the opposite is true -- without the gate, flipped_face_pct is
   15-33% LOWER and self-intersections 60-63% lower. What the gate actually buys
   is transit (-42% lh / -37% rh) and end-in-WM (-71% / -70%), at the cost of
   mesh quality. Magnified coronal views of the deepest sulci show why: the
   gated pial descends into the sulcal fundus along the CSF line, while the
   ungated one bridges across the opening and stays on the gyral envelope. A
   surface that never enters a sulcus has fewer chances to fold or to collide
   with the far bank, so its better mesh metrics reflect avoided geometry rather
   than better handling of it. Entering sulci is the correct behaviour, so the
   gate stays on by default -- and `transit` is flagging exactly the bridging it
   is meant to catch, which is a point in its favour as a proxy. Difference is
   global (88% of vertices move >0.1mm, mean 0.33mm) but concentrated in fundi.
   One subject.

2. `smoothing_sigma` — the
   *direction* of the velocity field comes from the gradient of a
   Gaussian-smoothed warped-WM-probability image (see
   `direct_cuda.gaussian_gradient_3d`), which is a second, independent
   smoothing operation from the one in (1). At the default sigma its
   9-voxel-wide derivative stencil spans a typical 1-2mm gyral blade and
   mixes both banks' directions. Narrowing it resolves direction locally
   without materially destabilising the solve (tested down to sigma=0.35;
   sigma=0.2 hits the solver's `grad_mag > 1e-3` cutoff and stalls).

   NOTE: 0.35 was the tested-best value for surfaces, but the DEFAULT here
   is now 1.0 -- the shipped solver's own default -- so that the thickness
   map this pipeline produces (via --write-thickness) stays comparable with
   stock DiReCT's. Pass --sigma 0.35 to recover the surface-tuned operating
   point. At 1.0, two differences from a stock DiReCT solve remain: the
   normal-gated velocity smoothing in (1), and the sulcal CSF sheet in (3),
   which alters the gm/wm probabilities fed to the solve.

3. `detect_sulcal_csf_sheet()` — a purely geometric, non-circular repair
   for unresolved sulci: from every GM voxel adjacent to WM, march outward
   along the interface normal; if the ray hits WM again before reaching
   background, the two banks have merged with no CSF voxel between them.
   Insert a fractional CSF marker at the local minimum of GM+WM evidence
   along that ray (not the geometric midpoint — the intensity minimum is
   measurably closer to ground truth). "Fractional" means the voxel's
   probabilities are weakened but its hard segmentation label is left
   alone, so no physical gap is carved and sulci are not visibly widened.

   STILL PAYING ITS KEEP. Ablated (--no-sulcal-sheet, everything else default)
   on the development subject: without it transit rises 6.1%/5.8%, crossed_csf
   8.5%/7.0%, flipped_face_pct 16.6%/12.0% and self-intersections 20.5%/15.8%,
   with config_loss 7.5%/6.6% worse. Unlike the normal gate in (1), the mesh
   metrics move the SAME way as transit, so there is no avoided-geometry trade
   to unpick -- the surface is simply better with it. The effect is also
   confined to where the sheet acts: vertices within 2 voxels of one of its
   5851 voxels are 15% of the surface but take 81% of the movement above 0.2mm,
   and move 8.8x further than the rest (0.24mm vs 0.03mm), identically on both
   hemispheres. Not verified: that the marked voxels really are merged sulcal
   banks rather than false positives -- only that the effect is localised to
   them and improves every metric. One subject.

Two further defects are propagation-side, not field-side:

4. `build_constrained_white()` — roughly half of any DiReCT-derived white
   surface's vertices sit, by construction, inside the WM label (the
   surface straddles the boundary). The solver's plain velocity is
   *identically zero* more than one voxel inside WM (see
   `direct_cuda.py`'s `active_mask = gm_mask + wm_contour`), so vertices
   that start there cannot move under propagation alone. This applies a
   Taubin-style relaxation to the starting surface with a hard
   signed-distance floor enforced every iteration, so no vertex is left
   (or placed) more than `floor` mm inside WM, while keeping the
   correction local enough not to fold the mesh. IMPORTANT: the Taubin
   lambda here (0.51) was empirically matched against the real pipeline's
   own `pymeshlab.apply_coord_taubin_smoothing` (see
   `dl_wm_surface_parallel_dev.py`, called with `stepsmoothnum=50`) to
   within 0.06mm mean vertex error — an earlier draft used lambda=0.3,
   which is measurably wrong (1.8mm mean error) and should not be reused.

   SUPERSEDED AS A STARTING-SURFACE REPAIR (default is now
   use_constrained_start=False). The premise above -- that vertices deep in
   WM cannot move because the velocity is zero there -- is already handled by
   mode='best', which samples the field at `pos + gu*(-dd + 0.25)` for in-WM
   vertices, i.e. just outside the boundary. Ablated on the development
   subject: the two pials differ by 0.038mm mean (median 0.013, 92% of
   vertices within 0.1mm), so the repair barely changes where propagation
   lands. Where it does differ it is mostly worse -- among pinned vertices,
   170 lh / 201 rh are >2x rougher WITH the repair against 86 / 114 without,
   and the repair leaves 14 lh pinned vertices above roughness 1.0 against 0
   without. Measured cause: white.corrected is ROUGHER than the raw white it
   is built from (0.079 vs 0.070 on pinned vertices, 0.107 vs 0.086 on free),
   because the hard floor projection ends each iteration after the Taubin
   pass. It remains available via the config field, and the same function is
   still used per-round inside propagate() to constrain the moving pial --
   that use is unaffected. One countervailing datum: without the repair the
   extreme free tail is worse (47 vs 31 lh vertices above roughness 1.0).

5. `propagate()` (mode='best') — a vertex whose *own* field points inward
   is projected onto its (mesh-smoothed) normal direction while it
   remains inside WM, then handed back to the raw field the instant it
   clears the boundary. An earlier version kept applying the escape rule
   even once outside WM, where the correction distance collapses to zero
   and 85% of the escape events silently froze the vertex in place for
   the rest of propagation — that bug is what motivated this rewrite.
   The escaped/field-following steps are then interleaved with rounds of
   the same constrained relaxation as (4), which keeps the mesh coherent
   without materially changing total displacement (tested with 2-20
   rounds spanning the same total path length: final mesh quality is
   governed by *total accumulated displacement*, not how finely it is
   subdivided — 10 rounds, matching the solver's own
   `num_integration_points`, is a reasonable default with no measured
   benefit from finer subdivision).

What is *not* fixed: a residual population of vertices where the field is
genuinely tangential to the true pial surface over an extended path (no
thin blade, no zero-field region — just a field that doesn't point where
the cortex is). No propagation-side or field-side change tested moved
this population; it likely needs a different solver-side treatment
(e.g. a boundary condition that is aware of the true GM/CSF interface,
not just the GM/WM one) that is out of scope here.

Coordinate frames
------------------
This module builds the voxel<->tkrRAS transform from the reference image's
own header (the cropped `seg_<Label>.nii.gz` grid), via the same
`get_vox2ras_tkr()` construction the pipeline's own surface-generation code
uses (dl_wm_surface_parallel_dev.py). The white surface it reads must
therefore be in the tkrRAS of THAT grid, which is what
`preparedata.py --space cropped` + dl_wm_surface_parallel_dev.py produce
(see doc/cropped-space-pipeline.md).

CORRECTION. An earlier version of this note claimed that sampling on the
cropped grid while the surfaces were built on the conformed 256^3 grid is
correct because the two centrings "cancel exactly". They do not, and the
0.79 mm frame-check distance quoted as evidence was itself the symptom.
`conform()` maps cropped voxel (S-1)//2 onto conformed voxel 127 (an
integer shift; measured (62,57,44) on sub-POBHC0002, lossless), whereas
tkrRAS centres on S/2. For an odd crop dimension the two grids' tkr frames
differ by exactly half a voxel. Measured on that subject (131x141x167, all
odd): tkr_conformed - tkr_cropped = (+0.5, -0.5, +0.5) mm, so every white
vertex was placed at (-0.5, -0.5, -0.5) voxels from its true position
(0.866 vox), and `check_frame_alignment` read 0.85 vox mean |distance| to
the WM boundary where the correctly placed surface reads 0.48. Re-propagating
the same white from its correct position, same saved velocity field, moves
the pial by 0.50 mm mean (median 0.48, 94% of vertices by >0.1 mm) on both
hemispheres; numbers in doc/cropped-space-pipeline.md section 5.

`main()` now refuses a white surface whose embedded volume geometry does not
match the reference grid, and warns when the surface carries none (every
output written before volume_info was added). The tkr frame of a single
grid carries no link to scanner RAS; to compare with another package's
surfaces go through world coordinates (`surface_frames.our_surface_to_world`,
`dldirect.compare_surfaces`), never by translating tkr coordinates.

Always call `check_frame_alignment()` before trusting a result on a
*different* pipeline output. Use the distance it reports, not the WM
fraction — see that function's docstring for why the ~50% WM check that
earlier work here relied on cannot detect the failure it was meant to.
Its 1.5 threshold is a gross-misplacement guard: it passed the half-voxel
error above.

Caveats
-------
- Investigated on a single subject, single hemisphere. No claim is made
  that the operating points below (tested-best sigma=0.35 -- see note in
  (2) on why the default is 1.0 -- dip<=0.95 sheet, floor=-0.5)
  transfer to other data without re-checking.
- `evaluate_surface()` reports only self-consistency metrics (does the
  path cross tissue it shouldn't, is the mesh manifold) since agreement
  with a reference pial surface requires subject-specific ground truth
  this module has no general way to obtain.
"""
import argparse
import dataclasses
import os
import sys

import numpy as np
import nibabel as nib
import pandas as pd
import scipy.special as ss
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import trimesh
from scipy import spatial
from scipy.ndimage import distance_transform_edt, map_coordinates, binary_dilation

from .direct_cuda import kelly_kapowski_cuda, gaussian_smooth_3d
from .surface_frames import volume_info_from_prep
from . import direct_cuda as _direct_cuda_module


# ---------------------------------------------------------------------------
# Coordinate frame (shared with dl_wm_surface_parallel_dev.py)
# ---------------------------------------------------------------------------

def get_vox2ras_tkr(img):
    """FreeSurfer-style tkrRAS transform, built from an image's own header.
    Copied verbatim from dl_wm_surface_parallel_dev.py so this module's
    frame construction matches the one the pipeline's surfaces are built
    with, rather than re-deriving it."""
    ds = img.header._structarr['pixdim'][1:4]
    ns = img.header._structarr['dim'][1:4] * ds / 2.0
    return np.array([[-ds[0], 0, 0, ns[0]],
                      [0, 0, ds[2], -ns[2]],
                      [0, -ds[1], 0, ns[1]],
                      [0, 0, 0, 1]], dtype=np.float64)


def make_transforms(ref_img):
    """Return (tovox, totkr): callables mapping Nx3 point arrays between
    this image's voxel-index space and its tkrRAS space."""
    A = get_vox2ras_tkr(ref_img)
    Ainv = np.linalg.inv(A)
    tovox = lambda p: nib.affines.apply_affine(Ainv, p)
    totkr = lambda q: nib.affines.apply_affine(A, q)
    return tovox, totkr


def check_frame_alignment(white_verts, seg, tovox, wm_label=3):
    """Sanity check that the reference image and the surface share a grid.

    Returns (mean_abs_distance_mm, wm_fraction, in_bounds_fraction).

    The test that matters is the first one: a white surface lies *on* the
    WM/GM interface, so the distance from its vertices to the WM boundary
    should be a small fraction of a voxel. Measured on a correct run:
    0.79 mm mean, 82% of vertices within 1 mm. Deliberately mis-registering
    the same surface by 13.5 mm raises it to 2.81 mm with 16% within 1 mm,
    so this quantity separates the two cleanly.

    The WM fraction is returned for continuity with earlier work but must
    NOT be used as the check. A valid frame and a badly wrong one both give
    roughly 50%, because brain tissue is roughly 50/45 WM/GM everywhere and
    a displaced surface still samples brain. It cannot detect the failure
    it was being relied on to detect.
    """
    wm_mask = (seg == wm_label)
    sd = distance_transform_edt(~wm_mask) - distance_transform_edt(wm_mask)

    vox = tovox(white_verts)
    ok = ((vox >= 0) & (vox < np.array(seg.shape) - 1)).all(1)
    dist = map_coordinates(sd, vox[ok].T, order=1)

    idx = np.rint(vox[ok]).astype(int)
    frac_wm = (seg[tuple(idx.T)] == wm_label).mean()
    return float(np.abs(dist).mean()), float(frac_wm), float(ok.mean())


# ---------------------------------------------------------------------------
# Input construction (mirrors DiReCT.py's own seg/gmT/wmT logic)
# ---------------------------------------------------------------------------

def load_gm_wm_probability(prep_dir, gm_labels=None, wm_labels=None):
    """Load per-label DeepSCAN logits from `prep_dir` (the same
    `seg_<Label>.nii.gz` files DiReCT.py reads) and collapse them into
    combined GM / WM probability maps.

    `gm_labels` defaults to cortex only (Left/Right-Cerebral-Cortex) —
    deliberately narrower than DiReCT.py's shipped default, which also
    includes the amygdala and hippocampus and was found in this
    investigation to let flow leak into subcortical grey matter.
    """
    if gm_labels is None:
        gm_labels = ['Left-Cerebral-Cortex', 'Right-Cerebral-Cortex']
    if wm_labels is None:
        wm_labels = ['Left-Cerebral-White-Matter', 'Right-Cerebral-White-Matter']
        hypo = os.path.join(prep_dir, 'seg_WM-hypointensities.nii.gz')
        if os.path.exists(hypo):
            wm_labels = wm_labels + ['WM-hypointensities']

    def _load(label):
        return nib.load(os.path.join(prep_dir, 'seg_%s.nii.gz' % label)).get_fdata(dtype=np.float32)

    ref_img = nib.load(os.path.join(prep_dir, 'seg_%s.nii.gz' % gm_labels[0]))
    gm_logit = np.max(np.stack([_load(l) for l in gm_labels]), axis=0)
    wm_logit = np.max(np.stack([_load(l) for l in wm_labels]), axis=0)
    gm_prob = np.where(gm_logit == 0, 0, ss.expit(gm_logit))
    wm_prob = np.where(wm_logit == 0, 0, ss.expit(wm_logit))
    return gm_prob.astype(np.float32), wm_prob.astype(np.float32), ref_img


def save_like_direct(arr, path, ref_img):
    """Write a volume exactly as DiReCT.py's save_img does (reference affine,
    mm units), so the files this module writes are drop-in replacements for
    the ones DiReCT.py would have written to the same directory."""
    img = nib.Nifti1Image(arr, ref_img.affine)
    img.header['xyzt_units'] = 2  # mm
    nib.save(img, path)


def build_seg_maps(gm_prob, wm_prob, hole_thr=0.7):
    """Reproduces DiReCT.py's seg/gmprobT/wmprobT construction exactly:
    argmax label, background where combined evidence is weak, then
    binarise everywhere except the residual band where gm<=0.5 and
    gm>=wm (found in this investigation to be inert — see module
    docstring's fix #2/#3, which are the changes that actually matter)."""
    seg = (np.argmax([gm_prob, wm_prob], axis=0) + 2).astype(np.uint8)
    seg[gm_prob + wm_prob < hole_thr] = 0
    seg[gm_prob > 0.5] = 2
    seg[wm_prob > 0.5] = 3

    gmT = gm_prob.copy()
    wmT = wm_prob.copy()
    gmT[wmT > gmT] = 0
    wmT[wmT > gmT] = 1
    wmT[seg < 1] = 0
    gmT[gmT > 0.5] = 1
    wmT[gmT > 0.5] = 0
    return seg, gmT, wmT




def rasterize_mesh_pv(verts_vox, faces, shape, supersample=3):
    """Fractional (partial-volume) occupancy of a closed mesh, per voxel.

    Rasterizes on a `supersample`-times finer grid and averages down, so a
    boundary voxel carries the fraction of its volume inside the surface rather
    than a 0/1 staircase. An original voxel i spans [i-0.5, i+0.5), so sub-voxel
    j maps back to (j+0.5)/S - 0.5, which is the shift applied here.

    Measured on sub-POBHC0002_TI900TR1700_run1 (131x141x167, both hemispheres):
    S=2 2.8 s, S=3 4.8 s, S=4 6.8 s; partial voxels 132k / 177k / 200k. Total
    volume converges -- crisp 525680, S=2 525124, S=3 525064, S=4 525005 -- so
    crisp rasterization is nearly unbiased in volume (-0.117%) and what the
    supersampling buys is sub-voxel boundary placement, not a different WM
    volume. S=3 is the default: S=4 moves the volume a further 0.011% for 40%
    more time and 2.4x the memory.
    """
    S = int(supersample)
    if S <= 1:
        return rasterize_mesh(verts_vox, faces, shape).astype(np.float32)
    ss = tuple(int(x) * S for x in shape)
    v = (np.asarray(verts_vox, np.float64) + 0.5) * S - 0.5
    m = rasterize_mesh(v, faces, ss)
    return m.reshape(shape[0], S, shape[1], S, shape[2], S).mean(axis=(1, 3, 5),
                                                                 dtype=np.float32)


def reconcile_seg_with_surface(seg, gmT, wmT, wm_mask, label_mask=None):
    """Replace the segmentation's white matter with the white SURFACE's interior.

    The velocity field is solved on `seg`, but the propagation starts from the
    white surface, and the two disagree: `seg == 3` is a threshold on the WM
    logits while the surface is a topology-corrected level set of a different
    mask. The field therefore has its inner boundary in a slightly different
    place from the surface that rides it, and no floor applied afterwards can
    repair that -- measured: pointing only the floor at the surface leaves the
    penetration unchanged (measured before this option was removed).

    WM becomes exactly `wm_mask` (rasterize_mesh of the white surfaces):
      * WM in `seg` but OUTSIDE the surface -> cortex, because the surface is
        the definition of where cortex begins;
      * inside the surface but not WM in `seg` (ventricles, subcortical grey,
        the corpus callosum fill) -> WM, since they are interior to the white
        surface and so are not cortex the thickness solve should flow through.

    `gmT`/`wmT` are updated to match so the solve's priors do not contradict
    its labels. `gm_prob`/`wm_prob` are left alone: they are raw intensities
    used for sulcal-dip detection, not labels.

    `wm_mask` may be boolean (crisp) or a float occupancy in [0,1] from
    rasterize_mesh_pv, in which case labels come from the 0.5 threshold but the
    solve's priors carry the sub-voxel fraction.

    Returns (seg, gmT, wmT, n_demoted, n_promoted); inputs are not modified.
    """
    frac = np.asarray(wm_mask)
    partial = frac.dtype.kind == 'f' and ((frac > 0) & (frac < 1)).any()
    if label_mask is not None:
        # Labels from a mask of their own, so that resolving the boundary more
        # finely does not also move which voxels are called WM. Without this the
        # 0.5-occupancy threshold disagrees with "centre inside" on ~1400 voxels,
        # which shifts the sulcal-sheet detection (it reads seg) and confounds
        # any measurement of what the partial volume itself did.
        binary = np.asarray(label_mask, bool)
    else:
        binary = frac >= 0.5 if partial else np.asarray(wm_mask, bool)
    seg2, gm2, wm2 = seg.copy(), gmT.copy(), wmT.copy()
    demoted = (seg == 3) & ~binary
    promoted = binary & (seg != 3)
    seg2[demoted] = 2
    seg2[promoted] = 3
    if partial:
        # Sub-voxel version: WM occupancy IS the fraction, and total tissue is
        # conserved, so whatever leaves WM becomes cortex. This reduces to the
        # crisp assignment below wherever the fraction is 0 or 1 -- a boundary
        # voxel 0.4 inside the surface gets wm 0.4 / gm 0.6 instead of 1/0 or
        # 0/1, a ventricle voxel (no tissue evidence, fraction 1) gets wm 1 /
        # gm 0, and deep cortex (fraction 0) keeps its own gm evidence.
        total = np.clip(gmT + wmT, 0.0, 1.0)
        wm2 = frac.astype(wmT.dtype, copy=True)
        gm2 = np.clip(total - frac, 0.0, 1.0).astype(gmT.dtype)
    else:
        gm2[demoted] = 1.0
        wm2[demoted] = 0.0
        gm2[promoted] = 0.0
        wm2[promoted] = 1.0
    return seg2, gm2, wm2, int(demoted.sum()), int(promoted.sum())


# ---------------------------------------------------------------------------
# Sulcal CSF detection (fix #3)
# ---------------------------------------------------------------------------

def detect_sulcal_csf_sheet(seg, gm_prob, wm_prob, dip_threshold=0.95,
                             max_ray_mm=4.0, step_mm=0.25):
    """Geometric detector for unresolved sulci: GM voxels adjacent to WM
    whose outward ray (along the WM signed-distance gradient) hits WM
    again before reaching background. The insertion point is the local
    minimum of gm+wm evidence along the ray, not the geometric midpoint —
    verified to disagree with the midpoint on ~1/6 of voxels and to give
    a better transit/agreement trade at matched sheet size.

    `dip_threshold` requires the trough to fall to at most this fraction
    of the ray-endpoint evidence before a voxel is marked; `None` marks
    every blocked ray regardless of dip depth (the most aggressive,
    least discriminating setting). 0.95 was the best-performing choice
    tested (vs. 0.85 and unconstrained).

    Returns (sheet_mask, n_blocked_rays, dip_depths).
    """
    wmb = (seg == 3)
    sdt = distance_transform_edt(~wmb) - distance_transform_edt(wmb)
    grad = np.stack(np.gradient(sdt), axis=-1)
    nu = grad / np.maximum(np.linalg.norm(grad, axis=-1), 1e-9)[..., None]
    total_ev = (gm_prob + wm_prob).astype(np.float32)

    st = np.ones((3, 3, 3), bool)
    src = (seg == 2) & binary_dilation(wmb, structure=st)
    ii = np.array(np.nonzero(src)).T.astype(float)
    d = nu[src.nonzero()]
    radii = np.arange(step_mm, max_ray_mm + 1e-9, step_mm)

    seg_samples = np.stack(
        [map_coordinates(seg, (ii + d * r).T, order=0, mode='nearest') for r in radii], axis=1)
    ev_samples = np.stack(
        [map_coordinates(total_ev, (ii + d * r).T, order=1, mode='nearest') for r in radii], axis=1)

    hit = np.full(len(ii), -1)
    reached_bg = np.zeros(len(ii), bool)
    for j in range(len(radii)):
        reached_bg |= (seg_samples[:, j] == 0) & (hit < 0)
        new_hit = (seg_samples[:, j] == 3) & (hit < 0) & ~reached_bg
        hit[new_hit] = j
    blocked = (hit > 0) & ~reached_bg

    dip_depths = []
    insertion_points = []
    for i in np.where(blocked)[0]:
        span = ev_samples[i, :hit[i]]
        if len(span) < 2:
            continue
        j = int(np.argmin(span))
        endpoint_ev = min(span[0], span[-1])
        dip = span[j] / max(endpoint_ev, 1e-6)
        dip_depths.append(dip)
        if dip_threshold is not None and dip > dip_threshold:
            continue
        insertion_points.append(ii[i] + d[i] * radii[j])

    mask = np.zeros(seg.shape, bool)
    if insertion_points:
        q = np.rint(np.array(insertion_points)).astype(int)
        valid = ((q >= 0) & (q < np.array(seg.shape))).all(1)
        q = q[valid]
        mask[q[:, 0], q[:, 1], q[:, 2]] = True
    mask &= (seg == 2)
    return mask, int(blocked.sum()), np.array(dip_depths)


def apply_fractional_sheet(gm_prob, wm_prob, sheet_mask, frac=0.0):
    """Weaken GM/WM evidence at sheet voxels WITHOUT changing their
    segmentation label — the voxel stays in the flow domain, it just
    stops reading as confident tissue. This is what avoids visibly
    widening sulci (a hard `seg=0` sheet was tested and rejected for
    exactly that reason, despite scoring marginally better numerically)."""
    gm2, wm2 = gm_prob.copy(), wm_prob.copy()
    gm2[sheet_mask] *= frac
    wm2[sheet_mask] *= frac
    return gm2, wm2


# ---------------------------------------------------------------------------
# Normal-gated velocity smoothing (fix #1)
# ---------------------------------------------------------------------------

class NormalGate:
    """Context manager that monkeypatches
    dldirect.direct_cuda.selective_masked_smooth_3d for the duration of a
    solve, replacing the shipped direction gate (which tests the field's
    own direction — unreliable exactly where it matters) with a gate on
    the WM interface normal (purely geometric)."""

    def __init__(self, seg, sigma_scale=2.0, coherence_threshold=0.7, mode='soft'):
        self.seg = seg
        self.sigma_scale = sigma_scale
        self.coherence_threshold = coherence_threshold
        self.mode = mode
        self._orig = None
        self._bipolar_cache = {}
        self._nu_t = None

    def __enter__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        wmb = (self.seg == 3)
        sdt = distance_transform_edt(~wmb) - distance_transform_edt(wmb)
        grad = np.stack(np.gradient(sdt), axis=-1)
        nu = grad / np.maximum(np.linalg.norm(grad, axis=-1), 1e-9)[..., None]
        self._nu_t = torch.from_numpy(nu.transpose(3, 0, 1, 2)[None].astype(np.float32)).to(device)

        def normal_masked_smooth(vol, sigma, dev, truncate=2.0, mode='hard'):
            r = max(1, int(truncate * sigma + 0.5))
            coords = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
            g1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
            g1d = g1d / g1d.sum()
            D, H, W = vol.shape[2:]
            padded = F.pad(vol, (r, r, r, r, r, r), mode='replicate')
            padded_nu = F.pad(self._nu_t, (r, r, r, r, r, r), mode='replicate')
            acc = torch.zeros_like(vol)
            wacc = torch.zeros((vol.shape[0], 1, D, H, W), device=dev)
            for dz in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        wgt = float(g1d[dz + r] * g1d[dy + r] * g1d[dx + r])
                        if wgt < 1e-6:
                            continue
                        sh = padded[:, :, r + dz:r + dz + D, r + dy:r + dy + H, r + dx:r + dx + W]
                        sn = padded_nu[:, :, r + dz:r + dz + D, r + dy:r + dy + H, r + dx:r + dx + W]
                        dot = (sn * self._nu_t).sum(dim=1, keepdim=True)
                        dw = (dot > 0).to(vol.dtype) if mode == 'hard' else ((1.0 + dot) * 0.5)
                        acc = acc + wgt * dw * sh
                        wacc = wacc + wgt * dw
            return torch.where(wacc > 1e-6, acc / wacc.clamp(min=1e-6), vol)

        def selective_normal(vol, sigma, dev, truncate=2.0, mode='hard',
                              coherence_threshold=0.5, gate='coherence', return_stats=False):
            plain = gaussian_smooth_3d(vol, sigma, dev, zero_boundary=False, truncate=truncate)
            key = round(float(sigma), 4)
            if key not in self._bipolar_cache:
                pn = gaussian_smooth_3d(self._nu_t, sigma, dev, zero_boundary=False, truncate=truncate)
                mag = (self._nu_t ** 2).sum(dim=1, keepdim=True).sqrt()
                smooth_mag = gaussian_smooth_3d(mag, sigma, dev, zero_boundary=False, truncate=truncate)
                coh = (pn ** 2).sum(dim=1, keepdim=True).sqrt() / smooth_mag.clamp(min=1e-6)
                self._bipolar_cache[key] = (coh < self.coherence_threshold) & (smooth_mag > 1e-6)
            bipolar = self._bipolar_cache[key]
            masked = normal_masked_smooth(vol, sigma * self.sigma_scale, dev,
                                           truncate=truncate / self.sigma_scale, mode=self.mode)
            out = torch.where(bipolar, masked, plain)
            return (out, float(bipolar.float().mean())) if return_stats else out

        self._orig = _direct_cuda_module.selective_masked_smooth_3d
        _direct_cuda_module.selective_masked_smooth_3d = selective_normal
        return self

    def __exit__(self, *exc):
        _direct_cuda_module.selective_masked_smooth_3d = self._orig
        return False


# ---------------------------------------------------------------------------
# Solve (fix #1 + #2 combined via the NormalGate context manager)
# ---------------------------------------------------------------------------

def solve_velocity_field(seg, gm_prob, wm_prob, ref_img, out_prefix,
                          smoothing_sigma=1.0, gradient_sigma=None, use_normal_gate=True,
                          num_integration_points=None, gradient_gate=None,
                          gate_sigma_scale=2.0, gate_coherence_threshold=0.7,
                          gate_mode='soft', verbose=False):
    """Solve for the DiReCT velocity field. With `use_normal_gate=True`
    (the tested-best configuration) this installs NormalGate for the
    duration of the solve and requests the solver's masked-smoothing path;
    with False, this is a call with the solver's own defaults (the "naive"
    comparison point). `gate_sigma_scale`/`gate_coherence_threshold`/
    `gate_mode` are exposed for parameter search — see field_pial_optimize.py;
    the tested-best values are the defaults here."""
    # Only Velocity.nii.gz is read back (below); the cumulative Forward/Inverse
    # fields are never touched by the propagation and cost more than the solve.
    kwargs = dict(verbose=verbose, smoothing_sigma=smoothing_sigma,
                   velocity_field_prefix=out_prefix, ref_img=ref_img,
                   cumulative_fields=False, return_velocity=True)
    if gradient_gate is not None:
        kwargs['gradient_gate'] = gradient_gate
    if num_integration_points is not None:
        # The saved Velocity.nii.gz is a PER-INTEGRATION-POINT field: the solve
        # composes it num_integration_points times. propagate() applies it once
        # per round, so the matched propagation_rounds equals this value.
        kwargs['num_integration_points'] = num_integration_points
    if gradient_sigma is not None:
        kwargs['gradient_sigma'] = gradient_sigma
    if use_normal_gate:
        kwargs.update(velocity_smooth_selective=gate_coherence_threshold,
                       velocity_smooth_mask_mode=gate_mode)
        with NormalGate(seg, sigma_scale=gate_sigma_scale,
                         coherence_threshold=gate_coherence_threshold, mode=gate_mode):
            thickness, velocity = kelly_kapowski_cuda(seg, gm_prob, wm_prob, **kwargs)
    else:
        thickness, velocity = kelly_kapowski_cuda(seg, gm_prob, wm_prob, **kwargs)
    if velocity is None:  # no prefix given, or an older solver: fall back to the file
        velocity = np.asarray(nib.load(out_prefix + 'Velocity.nii.gz').dataobj)
    return velocity, thickness


# ---------------------------------------------------------------------------
# Starting-surface repair (fix #4)
# ---------------------------------------------------------------------------

def _mesh_adjacency(vertices, faces):
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    nbrs = mesh.vertex_neighbors
    rows = np.concatenate([np.full(len(n), i) for i, n in enumerate(nbrs)])
    cols = np.concatenate([np.asarray(n) for n in nbrs])
    Wm = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(vertices), len(vertices)))
    deg = np.asarray(Wm.sum(1)).ravel()
    deg[deg == 0] = 1
    return mesh, Wm, deg


def build_constrained_white(white_verts, faces, seg, tovox, totkr,
                             floor=-0.5, iters=8, lam=0.51, cache=None, edges=None,
                             edge_ts=(0.5,), face_floor=False):
    """Taubin-style relaxation of the white surface with a hard
    signed-distance floor enforced every iteration, so vertices are
    smoothed *and* prevented from sitting in the zero-velocity interior
    of WM. `lam=0.51` is empirically matched to pymeshlab's
    apply_coord_taubin_smoothing defaults (see module docstring) — do
    not change without re-validating against that reference."""
    # `cache` supplies (Wm, deg, sdt, gu) -- all functions of `faces` and `seg`
    # only, so they are identical on every call. Recomputing them here cost
    # 2.1s per call (mesh adjacency 1.28s, the two distance transforms 0.67s,
    # the gradient 0.17s); propagate() calls this once per round, so ~21s of a
    # 27s per-hemisphere propagation was rebuilding the same arrays ten times.
    if cache is not None:
        Wm, deg, sdt, gu = cache
    else:
        _mesh, Wm, deg = _mesh_adjacency(white_verts, faces)
        wmb = (seg == 3)
        sdt = distance_transform_edt(~wmb) - distance_transform_edt(wmb)
        grad = np.stack(np.gradient(sdt), axis=-1)
        gu = grad / np.maximum(np.linalg.norm(grad, axis=-1), 1e-9)[..., None]
    sd_vox = lambda pv: map_coordinates(sdt, pv.T, order=1, mode='nearest')
    gu_vox = lambda pv: np.stack([map_coordinates(gu[..., k], pv.T, order=1, mode='nearest')
                                   for k in range(3)], axis=1)

    pos = tovox(white_verts).copy()
    # `floor` may be a scalar or a per-vertex array (see propagate's no_push);
    # broadcasting once here keeps the loop identical for both cases.
    fl = np.full(len(white_verts), float(floor)) if np.isscalar(floor) else np.asarray(floor, float)
    for i in range(iters):
        neighbour_mean = (Wm @ pos) / deg[:, None]
        mu = -0.53 if i % 2 else lam
        pos = pos + mu * (neighbour_mean - pos)
        sd = sd_vox(pos)
        over = sd < fl
        if over.any():
            correction = fl[over] - sd[over]
            pos[over] = pos[over] + gu_vox(pos[over]) * correction[:, None]
        if edges is not None:
            # The vertex floor says nothing about what happens BETWEEN vertices:
            # an edge whose endpoints both sit at the floor can still bow through
            # WM. Measured on rh, the default configuration left 3 cortical edges
            # below sd 0 and 40 below 0.5 while no vertex was under either.
            # Push both endpoints out when their midpoint falls below the floor
            # they share (the min of the two, so a no_push edge is left alone).
            a, b = edges[:, 0], edges[:, 1]
            acc = np.zeros_like(pos); cnt = np.zeros(len(pos))
            for t in edge_ts:
                pt = pos[a] * (1.0 - t) + pos[b] * t
                # Interpolate the floor along the edge rather than taking the
                # minimum. min() gave a mixed cortical/no_push edge the
                # permissive -0.5 floor along its whole length, and measured on
                # the candidate surface 180 (lh) / 220 (rh) such edges went
                # negative, to -0.94. Interpolating grades the constraint across
                # the boundary the way pin_feather grades the pin.
                fle = fl[a] * (1.0 - t) + fl[b] * t
                sdm = sd_vox(pt)
                bad = sdm < fle
                if not bad.any():
                    continue
                # Displacing both endpoints by `corr` moves the point at
                # parameter t by exactly `corr`, for any t.
                corr = (fle[bad] - sdm[bad])[:, None] * gu_vox(pt[bad])
                np.add.at(acc, a[bad], corr); np.add.at(cnt, a[bad], 1)
                np.add.at(acc, b[bad], corr); np.add.at(cnt, b[bad], 1)
            if face_floor:
                tri = np.asarray(faces)
                cen = pos[tri].mean(1)
                sdc = sd_vox(cen)
                flf = fl[tri].mean(1)
                badf = sdc < flf
                if badf.any():
                    corr = (flf[badf] - sdc[badf])[:, None] * gu_vox(cen[badf])
                    for k in range(3):
                        np.add.at(acc, tri[badf, k], corr)
                        np.add.at(cnt, tri[badf, k], 1)
            m = cnt > 0
            if m.any():
                pos[m] = pos[m] + acc[m] / cnt[m][:, None]
    return totkr(pos)


def rasterize_mesh(verts_vox, faces, shape, eps=1e-9):
    """Voxel mask of the closed region enclosed by a triangle mesh.

    A voxel is True iff its CENTRE lies inside the mesh. Parity fill along +x:
    every triangle crossing of each (j,k) grid line is collected, sorted, and
    the integer centres between successive crossing pairs are filled. Unlike a
    surface voxelization followed by a hole fill, this adds no shell, so the
    mask's boundary sits on the mesh rather than half a voxel outside it.

    `verts_vox` must already be in voxel coordinates of `shape` (i.e. tovox
    applied). Grid lines with an odd number of crossings are grazing hits and
    are skipped, which is why the mesh must be closed; the topology-corrected
    white surface is.

    Costs ~0.2 s per hemisphere on a cropped grid, so it is cheap enough to do
    at startup. Used by --wm-from-surface, and by the pial-inside-white metric:
    seg==3 is a segmentation of the WM logits, and where the white surface sits
    outside it a pial vertex can satisfy the floor and still be inside the white
    surface -- which end_in_wm cannot see.
    """
    v = np.asarray(verts_vox, np.float64)
    tri = v[np.asarray(faces)]
    y, z = tri[:, :, 1], tri[:, :, 2]
    jlo = np.ceil(y.min(1) - eps).astype(np.int64); jhi = np.floor(y.max(1) + eps).astype(np.int64)
    klo = np.ceil(z.min(1) - eps).astype(np.int64); khi = np.floor(z.max(1) + eps).astype(np.int64)
    np.clip(jlo, 0, shape[1] - 1, out=jlo); np.clip(jhi, 0, shape[1] - 1, out=jhi)
    np.clip(klo, 0, shape[2] - 1, out=klo); np.clip(khi, 0, shape[2] - 1, out=khi)
    nj = np.maximum(jhi - jlo + 1, 0); nk = np.maximum(khi - klo + 1, 0)
    cnt = nj * nk
    keep = cnt > 0
    out = np.zeros(shape, bool)
    if not keep.any():
        return out
    ti = np.repeat(np.nonzero(keep)[0], cnt[keep])
    off = np.arange(len(ti)) - np.repeat(np.cumsum(cnt[keep]) - cnt[keep], cnt[keep])
    njk = np.repeat(nk[keep], cnt[keep])
    jj = np.repeat(jlo[keep], cnt[keep]) + off // njk
    kk = np.repeat(klo[keep], cnt[keep]) + off % njk
    a, b, c = tri[ti, 0], tri[ti, 1], tri[ti, 2]
    d = ((b[:, 1] - a[:, 1]) * (c[:, 2] - a[:, 2]) - (c[:, 1] - a[:, 1]) * (b[:, 2] - a[:, 2]))
    ok = np.abs(d) > eps
    py, pz = jj - a[:, 1], kk - a[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        w1 = (py * (c[:, 2] - a[:, 2]) - pz * (c[:, 1] - a[:, 1])) / d
        w2 = (pz * (b[:, 1] - a[:, 1]) - py * (b[:, 2] - a[:, 2])) / d
    inside = ok & (w1 >= -eps) & (w2 >= -eps) & (w1 + w2 <= 1 + eps)
    ti, jj, kk, w1, w2 = ti[inside], jj[inside], kk[inside], w1[inside], w2[inside]
    a, b, c = tri[ti, 0], tri[ti, 1], tri[ti, 2]
    xs = a[:, 0] + w1 * (b[:, 0] - a[:, 0]) + w2 * (c[:, 0] - a[:, 0])
    line = jj.astype(np.int64) * shape[2] + kk
    order = np.lexsort((xs, line))
    line, xs = line[order], xs[order]
    bnd = np.nonzero(np.r_[True, line[1:] != line[:-1], True])[0]
    for st, en in zip(bnd[:-1], bnd[1:]):
        x = xs[st:en]
        if len(x) % 2:
            continue
        j, k = divmod(int(line[st]), shape[2])
        for q in range(0, len(x), 2):
            i0 = int(np.ceil(x[q] - eps)); i1 = int(np.floor(x[q + 1] + eps))
            if i1 >= i0:
                out[max(i0, 0):min(i1, shape[0] - 1) + 1, j, k] = True
    return out


def build_no_push_mask(white_verts, faces, seg, soft_seg, id_map, tovox,
                       rings=0, radius=2):
    """Vertices with no cortex to move into: the pial should stay at the white
    surface there rather than be pushed outward into tissue that has no
    cortical ribbon over it.

    Two clauses:

      - the vertex sits on a structure with no cortex over it (hippocampus,
        amygdala, corpus callosum), or within `rings` MESH neighbours of one.
        `rings` defaults to 0: growing the region outward was measured to buy
        only ~25 end-in-WM vertices per hemisphere while raising flipped faces
        (0.157 -> 0.175 -> 0.189 as rings goes 0 -> 1 -> 2) and halving the
        travel of free parahippocampal vertices next to the block (rh 88357:
        2.32mm at rings=0 against 1.27mm at rings=2, versus 2.60mm unpinned);
      - the vertex has no cortical GM (seg == 2) within `radius` voxels, which
        is the deep-WM closure case.

    The structure clause expands along mesh edges, NOT by dilating the volume.
    A volumetric dilation crosses the gap between opposing banks of a sulcus and
    between neighbouring gyri -- voxel-adjacent but geodesically far -- so it
    flags vertices that belong to cortex somewhere else entirely. Ring expansion
    from vertices actually sitting on the structure cannot leave the surface.

    A vertex whose OWN label is a cortical parcel is never marked, whatever the
    neighbourhood says: it has cortex over it by definition. Without that guard
    the proximity test reached through the thin white-matter sheet separating
    hippocampus from parahippocampal cortex and pinned 22.4% of left
    parahippocampal and 19.1% of left entorhinal vertices to zero thickness --
    parcels extract_stats.py reports, and the ones that matter most in memory
    and dementia work.
    """
    names = ('Left-Hippocampus', 'Right-Hippocampus', 'Left-Amygdala',
             'Right-Amygdala', 'Corpus-Callosum')
    ids = [id_map[n] for n in names if n in id_map]
    idx = np.clip(np.rint(tovox(white_verts)).astype(int), 0,
                  np.array(seg.shape) - 1)
    i0, i1, i2 = idx[:, 0], idx[:, 1], idx[:, 2]
    own = soft_seg[i0, i1, i2]

    # seed on the structure itself, then grow along mesh edges only
    on_struct = np.isin(own, ids)
    _mesh, Wm, _deg = _mesh_adjacency(white_verts, faces)
    grown = on_struct.copy()
    for _ in range(max(rings, 0)):
        grown = grown | ((Wm @ grown.astype(np.int8)) > 0)

    near_gm = binary_dilation(seg == 2, iterations=radius)[i0, i1, i2]
    mask = grown | ~near_gm

    is_cortical = (own >= 1000) & (own < 3000)
    return mask & ~is_cortical


# ---------------------------------------------------------------------------
# Propagation (fix #5) and the naive comparison path
# ---------------------------------------------------------------------------

def smooth_retraction(traj, s, faces, Wm, deg, iters=10, lam=0.6,
                      max_iter=40, min_s=1.0, stats=None):
    """Remove the dimples a per-vertex retraction leaves, WITHOUT moving any
    vertex off its own trajectory.

    retract_self_intersections pulls individual vertices back along their paths,
    which leaves a vertex at s=8.9 sitting among neighbours still at s=10 --
    a spike in the retraction field, and so a dimple in the surface. Smoothing
    the FIELD rather than the positions fixes the shape while every vertex stays
    on the path the propagation took it along, so index correspondence with the
    white surface survives exactly.

    The smoothed field is clamped with `np.minimum` against the current one, so
    a vertex can only ever retract FURTHER. That turns each spike into a shallow
    basin (the neighbours come back to meet it) rather than filling it in, and
    means the surface only moves toward positions it already occupied earlier in
    the propagation. Intersections are re-checked afterwards and any that
    reappear are retracted away as usual.

    Returns (verts, s, info).
    """
    traj = np.asarray(traj, float)
    R = len(traj) - 1
    n = traj.shape[1]
    s = np.asarray(s, float).copy()
    min_s = float(np.clip(min_s, 0.0, R))

    def positions(s):
        lo = np.clip(np.floor(s).astype(int), 0, R - 1)
        frac = (s - lo)[:, None]
        idx = np.arange(n)
        return traj[lo, idx] * (1 - frac) + traj[lo + 1, idx] * frac

    for _ in range(iters):
        nb = (Wm @ s) / deg
        s = np.minimum(s, s + lam * (nb - s))

    verts = positions(s)
    n_after = _count_self_intersections(verts, faces)[0]
    extra = 0
    for _ in range(max_iter):
        sel = _self_intersecting_faces(verts, faces)
        if sel is None or not sel.any():
            break
        vsel = np.zeros(n, bool)
        vsel[np.asarray(faces)[sel].ravel()] = True
        if not (s[vsel] > floor_s[vsel]).any():
            break
        s[vsel] = np.maximum(s[vsel] - 0.25, floor_s[vsel])
        verts = positions(s)
        extra += 1
    info = dict(after_smoothing=n_after, extra_retractions=extra,
                final=_count_self_intersections(verts, faces)[0],
                mean_retraction=float((R - s[s < R]).mean()) if (s < R).any() else 0.0,
                retracted=int((s < R).sum()))
    if stats is not None:
        stats.update(info)
    return verts, s, info


def taubin_guarded(verts, faces, Wm, deg, iters=10, lam=0.51, mu=-0.53,
                   region=None, floor_fn=None, max_backtrack=6, verbose=False):
    """Taubin smoothing that provably cannot create a self-intersection.

    Each pass computes the usual Taubin displacement, then applies it with a
    per-vertex scale that starts at 1 and is HALVED on any vertex belonging to
    (or adjacent to) a face that the trial position would make intersect --
    repeating until the trial is clean, and zeroing the scale on stubborn
    vertices as a last resort. A vertex that cannot be smoothed safely simply
    is not smoothed, so the pass is a no-op there rather than a regression.

    Intended to run after retract_self_intersections, whose per-vertex pullback
    leaves a retracted vertex sitting behind its un-retracted neighbours.

    `region`: restrict smoothing to these vertices (e.g. the retracted set plus
    a halo); None smooths everything. `floor_fn` re-applies the signed-distance
    floor after each trial, so smoothing cannot push a vertex into WM.
    """
    cur = np.asarray(verts, float).copy()
    faces = np.asarray(faces)
    n = len(cur)
    for i in range(iters):
        step = lam if i % 2 == 0 else mu
        nb = (Wm @ cur) / deg[:, None]
        delta = step * (nb - cur)
        if region is not None:
            delta = delta * region[:, None]
        a = np.ones(n)

        def trial(a):
            c = cur + a[:, None] * delta
            if floor_fn is not None:
                c = floor_fn(c)
            # floor_fn (and any pin folded into it) moves vertices regardless of
            # `a`, so without this a scale of 0 would NOT be a no-op and the
            # backtrack could never recover the last known-clean position.
            zero = a == 0
            if zero.any():
                c[zero] = cur[zero]
            return c

        cand = trial(a)
        for _ in range(max_backtrack):
            sel = _self_intersecting_faces(cand, faces)
            if sel is None or not sel.any():
                break
            bad = np.zeros(n, bool)
            bad[faces[sel].ravel()] = True
            bad = (Wm @ bad.astype(np.float64)) > 0   # the whole offending ring backs off
            a[bad] *= 0.5
            cand = trial(a)
        else:
            sel = _self_intersecting_faces(cand, faces)
            if sel is not None and sel.any():
                bad = np.zeros(n, bool)
                bad[faces[sel].ravel()] = True
                bad = (Wm @ bad.astype(np.float64)) > 0
                a[bad] = 0.0
                cand = trial(a)
        cur = cand
        if verbose:
            print('    taubin pass %d: %d vertices held back'
                  % (i, int((a < 1).sum())), flush=True)
    return cur


def mesh_roughness(verts, Wm, deg):
    """Per-vertex distance from the mean of its neighbours -- the
    high-frequency content a smoothing pass would remove."""
    return np.linalg.norm(verts - (Wm @ verts) / deg[:, None], axis=1)


def _crossing_pairs(verts, faces, cand_faces, radius=None):
    """Exact triangle-triangle crossings among `cand_faces` (and their spatial
    neighbours), excluding pairs that share a vertex. Returns [(fa, fb), ...]."""
    verts = np.asarray(verts, float); faces = np.asarray(faces)
    cent = verts[faces].mean(1)
    if radius is None:
        e = np.linalg.norm(verts[faces[:, 1]] - verts[faces[:, 0]], axis=1)
        radius = float(np.percentile(e, 99) * 2.0)
    tree = spatial.cKDTree(cent)
    out = []
    seen = set()
    for a in np.asarray(cand_faces):
        for b in tree.query_ball_point(cent[a], radius):
            if b == a:
                continue
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            if set(faces[a]) & set(faces[b]):
                continue                       # shares a vertex: not a crossing
            if _tri_tri_hit(verts[faces[a]], verts[faces[b]]):
                out.append(key)
    return out


def _tri_tri_hit(A, B):
    """True if two triangles genuinely pass through one another."""
    for T, U in ((A, B), (B, A)):
        v0, v1, v2 = U
        e1 = v1 - v0; e2 = v2 - v0
        for i in range(3):
            o = T[i]; d = T[(i + 1) % 3] - T[i]
            p = np.cross(d, e2); det = e1 @ p
            if abs(det) < 1e-12:
                continue
            inv = 1.0 / det; t = o - v0; u = (t @ p) * inv
            if u < -1e-9 or u > 1 + 1e-9:
                continue
            q = np.cross(t, e1); v = (d @ q) * inv
            if v < -1e-9 or u + v > 1 + 1e-9:
                continue
            sN = (e2 @ q) * inv
            if -1e-9 <= sN <= 1 + 1e-9:
                return True
    return False


def untangle_by_separation(verts, faces, step=0.02, max_iter=200, max_move=0.5,
                           stats=None):
    """Resolve self-intersections with the SMALLEST movement that clears them.

    For each crossing pair the two triangles are pushed apart along the line
    joining their centroids, by `step` mm each, and the test is repeated. A
    crossing that is only 0.04-0.13 mm deep -- which is what these actually are
    -- therefore costs about that much movement, instead of the 3-4.5 mm that
    retract_self_intersections spends: retraction may only slide a vertex along
    its own trajectory, which is not the direction that separates two nearly
    coplanar triangles, so it has to undo most of the traverse to get there.

    `max_move` caps the total displacement of any vertex, so a pathological
    case fails visibly rather than by quietly deforming the surface.

    Returns (verts, info).
    """
    verts = np.asarray(verts, float).copy()
    faces = np.asarray(faces)
    start = verts.copy()
    n0, _ = _count_self_intersections(verts, faces)
    info = dict(initial=n0, iters=0, final=n0, moved=0, max_move=0.0, mean_move=0.0,
                capped=0)
    if n0 == 0:
        if stats is not None:
            stats.update(info)
        return verts, info
    for it in range(max_iter):
        sel = _self_intersecting_faces(verts, faces)
        if sel is None or not sel.any():
            break
        pairs = _crossing_pairs(verts, faces, np.where(sel)[0])
        if not pairs:
            break
        delta = np.zeros_like(verts)
        cnt = np.zeros(len(verts))
        for a, b in pairs:
            ca = verts[faces[a]].mean(0); cb = verts[faces[b]].mean(0)
            d = ca - cb
            nd = np.linalg.norm(d)
            if nd < 1e-9:                      # coincident centroids: use the normal
                d = np.cross(verts[faces[a][1]] - verts[faces[a][0]],
                             verts[faces[a][2]] - verts[faces[a][0]])
                nd = np.linalg.norm(d)
                if nd < 1e-12:
                    continue
            d = d / nd
            delta[faces[a]] += d * step; cnt[faces[a]] += 1
            delta[faces[b]] -= d * step; cnt[faces[b]] += 1
        act = cnt > 0
        if not act.any():
            break
        verts[act] += delta[act] / cnt[act][:, None]
        # enforce the displacement cap
        tot = verts - start
        mag = np.linalg.norm(tot, axis=1)
        over = mag > max_move
        if over.any():
            verts[over] = start[over] + tot[over] / mag[over][:, None] * max_move
            info['capped'] = int(over.sum())
        info['iters'] = it + 1
    mv = np.linalg.norm(verts - start, axis=1)
    info['final'] = _count_self_intersections(verts, faces)[0]
    info['moved'] = int((mv > 1e-9).sum())
    info['max_move'] = float(mv.max())
    info['mean_move'] = float(mv[mv > 1e-9].mean()) if (mv > 1e-9).any() else 0.0
    if stats is not None:
        stats.update(info)
    return verts, info


def retract_self_intersections(traj, faces, Wm=None, rings=0, step=0.5,
                               max_iter=60, min_s=0.0, max_move=0.05, stats=None):
    """Untangle by sliding each intersecting vertex BACK ALONG ITS OWN
    trajectory, rather than smoothing it sideways.

    `traj` is propagate(..., return_trajectory=True)'s (rounds+1, nverts, 3)
    stack. Every vertex carries an arc parameter `s` in [0, rounds]; its
    position is the piecewise-linear interpolation of its own path at `s`.
    Vertices belonging to an intersecting face have `s` reduced by `step`
    rounds and the test is repeated. A vertex therefore never leaves the path
    the propagation took it along -- index correspondence with the white
    surface is preserved exactly, and a fully retracted vertex (s=0) simply
    sits back on the white surface.

    This is the alternative to a soap-bubble repair, which resolves tangles
    with movement that is ~50% tangential to the traverse and so breaks that
    correspondence for the vertices it touches.

    `rings`: also retract this many mesh rings around each intersecting face
    (needs `Wm`); 0 retracts only the faces' own vertices.

    `max_move`: hard cap, in mm of travel back along the path, on how far any
    vertex may be retracted. THIS IS THE PRIMARY CONTROL and it is deliberately
    small (0.05 mm). The crossings being repaired are 0.04-0.13 mm deep, so a
    correction of that order is all one should cost; anything that needs more is
    not a tangle worth chasing but a degenerate patch of surface, and it is left
    in place. The residual `final` count is then a QUALITY SIGNAL -- surfaces
    with a high residue have bad geometry upstream, and should be reported
    rather than silently deformed until the metric reads zero.

    Without such a cap the loop runs away: retracting only the crossing vertices
    stretches the patch (their neighbours stay put) so the crossing persists,
    and the loop keeps pulling until it hits the floor -- measured on rh, 9
    vertices retracted 3.0-4.5 mm to clear a 0.13 mm overlap, and 21 vertices
    reached the white surface itself and were left with zero cortical thickness.

    Cap sweep, sub-POBHC0002_TI1100TR2600 (residual / fixed of ~2300 crossings,
    max movement mm, thickness delta mm, normal-reversal %, zero-thickness count):

      cap        rh                                lh
      0.02   1943/ 353  0.020  -0.0000 .0517 5210   1947/ 357  0.020 -0.0000 .0554 5896
      0.05   1458/ 838  0.050  -0.0000 .0517 5210   1475/ 829  0.050 -0.0001 .0551 5896
      0.10   1088/1208  0.100  -0.0001 .0520 5210   1071/1233  0.100 -0.0001 .0537 5897
      0.25    501/1795  0.250  -0.0002 .0513 5210    513/1791  0.250 -0.0003 .0537 5895
      1.00     55/2241  1.000  -0.0008 .0534 5210      7/2297  1.000 -0.0003 .0558 5895
      none      4/2292  5.220  -0.0039 .0548 5230      0/2304  1.811 -0.0003 .0558 5895

    98% of crossings are reachable within 1 mm; the whole 5 mm runaway is spent
    on the last ~50. Buying those costs 20 zero-thickness vertices and 5x the
    thickness error on rh -- and still does not reach zero (4 remain). At 0.05
    the repair is free on every measured axis: thickness -0.0000, reversals
    unchanged from unrepaired, no zero-thickness vertices added.

    Unrepaired reversal rates for reference: rh 0.0517, lh 0.0561.

    `min_s`: floor on the arc parameter, in rounds. Now secondary to `max_move`
    and defaulted to 0.0, since the mm cap binds first in every measured case.

    `rings`: also retract this many mesh rings around each crossing face, so the
    patch moves coherently instead of being stretched. Off by default.

    Returns (verts, s, info).
    """
    traj = np.asarray(traj, float)
    R = len(traj) - 1
    n = traj.shape[1]
    min_s = float(np.clip(min_s, 0.0, R))
    s = np.full(n, float(R))
    # Per-vertex arc parameter at which travel back from the end reaches
    # `max_move` mm. Path length, not straight-line: measured arc/chord is 1.00
    # on these trajectories, and it is monotone in s, which the cap needs.
    seg = np.linalg.norm(np.diff(traj, axis=0), axis=2)          # (R, n)
    s_cap = np.zeros(n)
    if max_move is not None:
        run = np.zeros(n); s_cap[:] = R
        remaining = np.full(n, float(max_move))
        for r in range(R - 1, -1, -1):
            L = seg[r]
            take = np.minimum(remaining, L)
            frac = np.where(L > 1e-12, take / np.maximum(L, 1e-12), 1.0)
            s_cap = np.where(remaining > 1e-12, (r + 1) - frac, s_cap)
            remaining = np.maximum(remaining - L, 0.0)
        s_cap = np.clip(s_cap, 0.0, R)
    floor_s = np.maximum(min_s, s_cap) if max_move is not None else np.full(n, min_s)

    def positions(s):
        lo = np.clip(np.floor(s).astype(int), 0, R - 1)
        frac = (s - lo)[:, None]
        idx = np.arange(n)
        return traj[lo, idx] * (1 - frac) + traj[lo + 1, idx] * frac

    verts = positions(s)
    n0, _ = _count_self_intersections(verts, faces)
    info = dict(initial=n0, iters=0, final=n0, retracted=0, mean_retraction=0.0)
    if n0 == 0:
        return verts, s, info

    for it in range(max_iter):
        sel = _self_intersecting_faces(verts, faces)
        if sel is None or not sel.any():
            break
        vsel = np.zeros(n, bool)
        vsel[np.asarray(faces)[sel].ravel()] = True
        for _ in range(rings):
            vsel = (Wm @ vsel.astype(np.float64)) > 0
        if not (s[vsel] > floor_s[vsel]).any():
            break                      # everything already at the retraction cap
        s[vsel] = np.maximum(s[vsel] - step, floor_s[vsel])
        verts = positions(s)
        info['iters'] = it + 1

    info['final'] = _count_self_intersections(verts, faces)[0]
    moved = s < R
    info['retracted'] = int(moved.sum())
    info['mean_retraction'] = float((R - s[moved]).mean()) if moved.any() else 0.0
    if stats is not None:
        stats.update(info)
    return verts, s, info


def _floor_project(verts, fl, sdt, gu, tovox, totkr):
    """Push any vertex below the signed-distance floor back out along the
    distance gradient. The projection half of build_constrained_white, without
    the smoothing -- used after a repair pass, which is itself a smoothing."""
    pos = tovox(verts).copy()
    sd = map_coordinates(sdt, pos.T, order=1, mode='nearest')
    over = sd < fl
    if over.any():
        g = np.stack([map_coordinates(gu[..., k], pos[over].T, order=1, mode='nearest')
                      for k in range(3)], axis=1)
        pos[over] = pos[over] + g * (fl[over] - sd[over])[:, None]
    return totkr(pos)


def repair_self_intersections(verts, faces, Wm, deg, rings=1, max_iter=40,
                              lam=0.6, floor_fn=None, stats=None):
    """Untangle self-intersecting faces by local smoothing, in the spirit of
    FreeSurfer's mris_remove_intersection: find the intersecting faces, smooth
    only their vertices (plus `rings` mesh rings around them), repeat.

    Local, so a surface with no intersections is returned untouched and the
    cost is proportional to the damage, not to the mesh. `floor_fn`, if given,
    is applied to the vertices after each smoothing pass to re-enforce the
    signed-distance floor -- without it the smoothing is free to pull vertices
    back into WM.

    `stats`: optional dict accumulated across calls, so a caller doing this
    once per round can total the work. Keys:
      calls          -- number of times this function ran
      faces_repaired -- sum over calls of the intersecting faces found on entry
      iters          -- total smoothing iterations
      unresolved     -- intersecting faces still present when max_iter ran out

    Returns (verts, n_initial, n_final).
    """
    verts = np.asarray(verts, float).copy()
    n0, _ = _count_self_intersections(verts, faces)
    if stats is not None:
        stats['calls'] = stats.get('calls', 0) + 1
        stats['faces_repaired'] = stats.get('faces_repaired', 0) + n0
        stats.setdefault('per_call', []).append(n0)
    if n0 == 0:
        return verts, 0, 0

    faces = np.asarray(faces)
    n = n0
    for it in range(max_iter):
        sel = _self_intersecting_faces(verts, faces)
        if sel is None or not sel.any():
            n = 0
            break
        vsel = np.zeros(len(verts), bool)
        vsel[faces[sel].ravel()] = True
        for _ in range(rings):
            vsel = (Wm @ vsel.astype(np.float64)) > 0
        neighbour_mean = (Wm @ verts) / deg[:, None]
        verts[vsel] = verts[vsel] + lam * (neighbour_mean[vsel] - verts[vsel])
        if floor_fn is not None:
            verts = floor_fn(verts)
        if stats is not None:
            stats['iters'] = stats.get('iters', 0) + 1
        n = int(sel.sum())
    else:
        n, _ = _count_self_intersections(verts, faces)
        if stats is not None:
            stats['unresolved'] = stats.get('unresolved', 0) + n
    return verts, n0, n


def _self_intersecting_faces(verts, faces):
    """Boolean mask over faces, or None if pymeshlab is unavailable. Unlike
    _count_self_intersections there is no proxy fallback -- a proximity count
    cannot say WHICH faces to smooth."""
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(np.asarray(verts, np.float64),
                                   np.asarray(faces, np.int32)))
        ms.compute_selection_by_self_intersections_per_face()
        return np.asarray(ms.current_mesh().face_selection_array(), bool)
    except Exception:
        return None


def _smoothed_normals(mesh, Wm, deg, k=2, lam=0.3):
    N = np.asarray(mesh.vertex_normals).copy()
    for _ in range(k):
        N = (1 - lam) * N + lam * (Wm @ N / deg[:, None])
        N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    return N


def build_pin_mask(no_push, white_verts, faces, seg, soft_seg, id_map, tovox,
                   scope='none', rings=0):
    """Subset of `no_push` to actually hold at the white surface.

      'none'        -- pin nothing (default). No-cortex vertices are still
                       exempt from the tightened floor; they drift ~0.14mm,
                       which is mesh-relaxation leakage from moving neighbours,
                       not the field (the velocity is exactly zero at ~90% of
                       them).
      'medial-wall' -- pin the corpus callosum and the midline/subcortical
                       closures (the no-GM clause: thalamus, brainstem,
                       ventricles, ventral DC, deep WM), but NOT the vertices
                       flagged by hippocampus/amygdala proximity. Those are the
                       ones whose rigid block roughens the surrounding pial:
                       measured exempt-region roughness 0.087 pinned against
                       0.071 unpinned (baseline 0.071).
      'all'         -- pin everything in `no_push`.
    """
    if scope == 'none':
        return np.zeros(len(white_verts), bool)
    if scope == 'all':
        return no_push
    if scope != 'medial-wall':
        raise ValueError("pin scope must be 'none', 'medial-wall' or 'all'")

    names = ('Left-Hippocampus', 'Right-Hippocampus',
             'Left-Amygdala', 'Right-Amygdala')
    ids = [id_map[n] for n in names if n in id_map]
    idx = np.clip(np.rint(tovox(white_verts)).astype(int), 0,
                  np.array(seg.shape) - 1)
    on_hipamy = np.isin(soft_seg[idx[:, 0], idx[:, 1], idx[:, 2]], ids)
    if rings > 0:
        _m, Wm, _d = _mesh_adjacency(white_verts, faces)
        for _ in range(rings):
            on_hipamy = on_hipamy | ((Wm @ on_hipamy.astype(np.int8)) > 0)
    return no_push & ~on_hipamy


def _pin_weights(no_push, Wm, feather):
    """Graded pin weight in [0,1] per vertex: 1 deep inside the no-cortex
    region, ramping to 0 at its boundary with free cortex.

    A hard pin puts the whole strain of a ~3mm displacement on the single ring
    of triangles spanning pinned and free vertices, which folds them (measured:
    ~20% of ring vertices land on flipped faces). Ramping over `feather` rings
    gives that seam somewhere to go while still holding the interior at zero.
    `feather=0` restores the hard pin.
    """
    n = len(no_push)
    if feather <= 0:
        return no_push.astype(np.float64)
    free = ~no_push
    dist = np.full(n, feather + 1, dtype=np.float64)
    dist[free] = 0.0
    reached = free.copy()
    frontier = free
    for k in range(1, feather + 1):
        nb = (Wm @ frontier.astype(np.int8)) > 0
        new = nb & ~reached
        if not new.any():
            break
        dist[new] = k
        reached |= new
        frontier = new
    return np.clip(dist / (feather + 1.0), 0.0, 1.0)


def propagate(white_verts, faces, velocity, seg, tovox, totkr, mode='best',
              rounds=10, smooth_iters=8, floor=-0.5, lam=0.51, use_escape=True,
              floor_after=None, floor_after_round=1, no_push=None,
              pin_mask=None, pin_feather=2, escape_in_no_push=False,
              step_scale=None, repair_from_round=None, repair_rings=1,
              repair_stats=None, return_trajectory=False,
              escape_on_magnitude=False, escape_full_clearance=False,
              escape_clearance=0.25, edge_floor=False, edge_ts=(0.5,),
              face_floor=False,
              retract_from_round=None, retract_step=0.25, retract_min_s=1.0,
              retract_rings=0, retract_stats=None):
    """mode='naive': single-pass, 10-step propagation with the raw
    (un-smoothed) vertex normals used only as a floor projection against
    inward motion — i.e. the shipped/obvious approach with none of this
    module's fixes.

    mode='best': escape-while-in-WM / follow-the-field-once-outside
    propagation (fix #5), using mesh-smoothed normals and an adaptive
    offset (sample the field just outside WM for vertices that start
    inside it, since the field is exactly zero more than ~1 voxel deep),
    interleaved with `rounds` applications of the constrained relaxation
    from `build_constrained_white`.

    `floor_after`: signed-distance floor for the per-round relaxation from
    round `floor_after_round` (0-based) onward; None keeps `floor` throughout.
    `no_push`: per-vertex bool marking vertices with no cortex to move into
    (hippocampus, midline, deep-WM closures) -- see build_no_push_mask.
    With `pin_no_push` (the default) these are held at their starting position
    for the whole propagation, so the pial coincides with the white there and
    the ribbon has zero thickness -- which is what the anatomy says, and what
    FreeSurfer does in the medial wall. With `pin_no_push=False` they are only
    exempted from the tightened `floor_after` and otherwise still follow the
    field. `pin_feather` ramps the pin over that many mesh rings (0 = hard pin).
    `escape_in_no_push=False` also suppresses the escape substitution inside the
    no-cortex region: escape exists to push a vertex out through cortex, and
    ~half of all escape activity was firing where there is no cortex to reach. Round 1 deliberately keeps
    the looser `floor` because that is where the escape substitution fires
    hardest -- tightening to 0.0 afterwards locks the escaped vertices outside
    WM instead of letting them drift back into the permitted band.

    `repair_from_round`: if set, run repair_self_intersections after the
    relaxation of every round from this one onward (0-based), so tangles are
    untangled as they form instead of once at the end. `repair_stats` collects
    the totals across rounds. None (the default) does no in-loop repair.

    `retract_from_round`: from this round onward, untangle after each round by
    sliding intersecting vertices BACK ALONG THE PATH SO FAR (the same operation
    as retract_self_intersections, applied to the partial trajectory) and then
    continuing to propagate from the retracted positions.

    MEASURED AND REJECTED -- kept so the experiment is not repeated. The premise
    was that, unlike the smoothing-based `repair_from_round`, this changes where
    the vertex samples the field next round and so could alter the trajectory
    rather than only erasing the symptom. It does change the sampling position;
    the surface re-tangles identically anyway. Per-round intersecting faces on
    rh, against the unrepaired run:

      unrepaired      25  104  21  13  12  19   26  114  686  2296
      from 0          25  104  18  26  11  24   37  110  680  2233
      from 3                       13  12  14   27  100  673  2235
      from 5                                19   28  100  675  2235
      from 7                                          114  658  2224

    Four starting points, four identical tails, every final count within 2.7% of
    doing nothing. Against post-hoc retract+smooth (int 0, thickness -0.0042,
    reversals 0.0517, end_in_wm 68, zero-thickness 5210, 47s), in-loop gives
    thickness -0.012 to -0.013, reversals 0.061-0.063, end_in_wm 70-80,
    zero-thickness 5401-5432 and 75-153s -- monotonically worse the earlier it
    starts. The zero-thickness excess is the mechanism: a vertex retracted at
    round 3 propagates from there and can be retracted again, accumulating at
    the `min_s` floor. From round 0 it exhausts the repair entirely -- a
    post-hoc pass afterwards still leaves 9 intersections it cannot fix.

    So the rounds 8-10 tangle is not accumulated damage that earlier
    intervention prevents; it is where the field takes the surface. Two
    structurally different in-loop repairs (sideways smoothing, on-path
    retraction) both leave the per-round curve unchanged.

    `retract_stats` collects the per-round work.

    `return_trajectory=True` additionally returns the (rounds+1, nverts, 3)
    stack of positions after each round, which retract_self_intersections needs
    to slide a vertex back along its own path.

    `edge_floor`: also enforce the signed-distance floor at EDGE MIDPOINTS, not
    just at vertices. The vertex-only constraint is silent about faces that bow
    through WM between two compliant vertices, which is what actually violates
    "the pial never enters white matter".

    `edge_ts`: where along each edge the floor is enforced (default midpoint
    only). `face_floor`: also enforce at face centroids. A bow peaking away from
    the midpoint escapes single-point enforcement entirely.

    `escape_on_magnitude`: also escape when the field step points outward but is
    too short to clear WM (the default triggers on direction alone).
    `escape_full_clearance`: move the full distance needed rather than capping at
    the field magnitude. Together these make escape able to guarantee that a
    vertex leaves WM; separately, neither can.

    `use_escape=False` keeps every other part of mode='best' -- the smoothed
    normals, the adaptive offset sampling, the per-round relaxation -- but
    drops the escape substitution, so a vertex inside WM whose field step
    points inward simply takes that inward step. Isolates what the escape
    contributes; not a validated configuration.
    """
    mesh, Wm, deg = _mesh_adjacency(white_verts, faces)
    wmb = (seg == 3)
    sdt = distance_transform_edt(~wmb) - distance_transform_edt(wmb)
    grad = np.stack(np.gradient(sdt), axis=-1)
    gu = grad / np.maximum(np.linalg.norm(grad, axis=-1), 1e-9)[..., None]
    relax_cache = (Wm, deg, sdt, gu)
    edge_arr = None
    if edge_floor:
        _f = np.asarray(faces)
        edge_arr = np.unique(np.sort(np.vstack([_f[:, [0, 1]], _f[:, [1, 2]],
                                                _f[:, [2, 0]]]), axis=1), axis=0)

    def sample_field_vox(pos_vox):
        # velocity is stored with (d,h,w) components matching voxel-index
        # order 1:1 (verified against a known-good reference in this
        # investigation, r=0.998, no permutation needed).
        return np.stack([map_coordinates(velocity[..., k], pos_vox.T, order=1, mode='nearest')
                          for k in range(3)], axis=1)

    def outward_step_tkr(pos_vox, sample_vox=None):
        # DiReCT's velocity convention points from GM into WM, so the
        # outward propagation direction is -v. totkr(pos-v) - totkr(pos)
        # applies just the affine's linear part to -v (translation
        # cancels), giving the step directly in tkrRAS mm without
        # needing the 3x3 submatrix exposed separately.
        v = sample_field_vox(sample_vox if sample_vox is not None else pos_vox)
        return totkr(pos_vox - v) - totkr(pos_vox)

    if mode == 'naive':
        N = np.asarray(mesh.vertex_normals)
        pos = tovox(white_verts)
        step0 = outward_step_tkr(pos)
        cos0 = np.einsum('ij,ij->i', step0, N) / np.maximum(np.linalg.norm(step0, axis=1), 1e-9)
        bad = cos0 < 0.0
        for _ in range(10):
            step_tkr = outward_step_tkr(pos)
            proj = np.einsum('ij,ij->i', step_tkr, N)
            fix = bad & (proj < 0)
            if fix.any():
                step_tkr[fix] = step_tkr[fix] - proj[fix, None] * N[fix]
            pos = tovox(totkr(pos) + step_tkr)
        return totkr(pos)

    # mode == 'best'
    N = _smoothed_normals(mesh, Wm, deg, k=2, lam=0.3)
    pinned_start = np.asarray(white_verts).copy()
    pin_w = None
    if pin_mask is not None and pin_mask.any():
        pin_w = _pin_weights(pin_mask, Wm, pin_feather)[:, None]
    cur = white_verts
    keep_traj = return_trajectory or retract_from_round is not None
    traj = [np.asarray(white_verts, float).copy()] if keep_traj else None
    for rnd in range(rounds):
        pos = tovox(cur)
        dd = map_coordinates(sdt, pos.T, order=1, mode='nearest')
        in_wm = dd < 0
        sample_pos = pos.copy()
        if in_wm.any():
            offset = np.zeros(len(pos))
            offset[in_wm] = -dd[in_wm] + 0.25
            nu_vox = np.stack([map_coordinates(gu[..., k], pos.T, order=1, mode='nearest')
                                for k in range(3)], axis=1)
            sample_pos = pos + nu_vox * offset[:, None]
        step_tkr = outward_step_tkr(pos, sample_vox=sample_pos)
        # Each round applies the whole field step, and rounds=10 matches the
        # solver's num_integration_points, so ten rounds composes the full
        # deformation. To take MORE, SHORTER steps the step must be scaled by
        # 10/rounds or the surface simply travels further.
        if step_scale is not None:
            step_tkr = step_tkr * step_scale
        mag = np.linalg.norm(step_tkr, axis=1)
        cos = np.einsum('ij,ij->i', step_tkr, N) / np.maximum(mag, 1e-9)
        need_all = np.maximum(-dd + escape_clearance, 0.0)
        if use_escape:
            escaping = (cos < 0.0) & in_wm
            if escape_on_magnitude:
                # Direction alone is not enough: a vertex whose field step points
                # correctly outward but is too SHORT to clear WM never escapes, so
                # it creeps and the round-`floor_after_round` floor projection has
                # to shove it instead -- measured on one hemisphere, 29.5% of
                # vertices were still inside WM entering round 1 and 4131 of them
                # took a >0.5mm floor kick, larger than a typical field step.
                outward = np.einsum('ij,ij->i', step_tkr, N)
                escaping = escaping | (in_wm & (outward < need_all))
        else:
            escaping = np.zeros(len(pos), bool)
        if no_push is not None and not escape_in_no_push:
            escaping = escaping & ~no_push
        if escaping.any():
            need = need_all[escaping]
            if escape_full_clearance:
                # Move the whole distance required to clear WM. The default caps
                # this at the field magnitude, which means escape cannot actually
                # guarantee clearance either.
                step_tkr[escaping] = need[:, None] * N[escaping]
            else:
                step_tkr[escaping] = np.minimum(mag[escaping], need)[:, None] * N[escaping]
        stepped = cur + step_tkr
        if floor_after is None or rnd < floor_after_round:
            round_floor = floor
        elif np.ndim(floor_after) > 0:
            round_floor = np.asarray(floor_after, float)   # per-vertex, e.g. feathered_floor
        elif no_push is None:
            round_floor = floor_after
        else:
            round_floor = np.where(no_push, floor, floor_after)
        cur = build_constrained_white(stepped, faces, seg, tovox, totkr,
                                       floor=round_floor, iters=smooth_iters, lam=lam,
                                       cache=relax_cache, edges=edge_arr,
                                       edge_ts=edge_ts, face_floor=face_floor)
        if pin_w is not None:
            # Re-pin after the relaxation, which would otherwise drag these
            # vertices along with their moving neighbours. Weighted, so the
            # boundary ring blends instead of stepping.
            cur = pin_w * pinned_start + (1.0 - pin_w) * cur
        if repair_from_round is not None and rnd >= repair_from_round:
            fl_arr = (np.full(len(cur), float(round_floor)) if np.isscalar(round_floor)
                      else np.asarray(round_floor, float))
            cur, _, _ = repair_self_intersections(
                cur, faces, Wm, deg, rings=repair_rings,
                floor_fn=lambda v: _floor_project(v, fl_arr, sdt, gu, tovox, totkr),
                stats=repair_stats)
            if pin_w is not None:
                cur = pin_w * pinned_start + (1.0 - pin_w) * cur
        if traj is not None:
            traj.append(np.asarray(cur, float).copy())
        if retract_from_round is not None and rnd >= retract_from_round:
            # Retract against the partial trajectory, then carry on propagating
            # from the pulled-back positions -- so the next round's field sample
            # is taken there, not at the tangled position.
            cur, _s, _info = retract_self_intersections(
                np.stack(traj), faces, Wm=Wm, rings=retract_rings,
                step=retract_step, min_s=min(retract_min_s, len(traj) - 1),
                max_iter=60)
            if pin_w is not None:
                cur = pin_w * pinned_start + (1.0 - pin_w) * cur
            traj[-1] = np.asarray(cur, float).copy()
            if retract_stats is not None:
                retract_stats.setdefault('per_round', []).append(_info['initial'])
                retract_stats['calls'] = retract_stats.get('calls', 0) + 1
                retract_stats['faces'] = retract_stats.get('faces', 0) + _info['initial']
                retract_stats['iters'] = retract_stats.get('iters', 0) + _info['iters']
    if return_trajectory:
        return cur, np.stack(traj)
    return cur


# ---------------------------------------------------------------------------
# Evaluation (no external ground truth required)
# ---------------------------------------------------------------------------

def _count_self_intersections(verts, faces):
    """(n_faces_intersecting, percent). Uses pymeshlab's exact face-face test.

    Falls back to the old vertex-proximity proxy if pymeshlab is unavailable,
    and says so in the returned percentage (-1.0) rather than silently
    substituting a different quantity.
    """
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(np.asarray(verts, np.float64),
                                   np.asarray(faces, np.int32)))
        ms.compute_selection_by_self_intersections_per_face()
        n = int(ms.current_mesh().selected_face_number())
        return n, 100.0 * n / max(len(faces), 1)
    except Exception:
        n = len(spatial.cKDTree(verts).query_pairs(0.15, output_type='ndarray'))
        return n, -1.0


def evaluate_surface(white_verts, pial_verts, faces, seg, tovox, n_samples=41,
                     no_push=None):
    """Self-consistency metrics only — see module docstring.

    `no_push`: per-vertex bool marking vertices deliberately held at the white
    surface (see build_no_push_mask). Those END in WM by construction -- that is
    the intended result, not a defect -- so `end_in_wm_count` is reported over
    the remaining vertices only, with the held ones in `end_in_wm_pinned`.
    Counting them as failures inflates config_loss by ~0.9-1.1 on this data,
    enough to rank a pinned run below doing nothing and to steer
    field_pial_optimize.py away from the pinning it should prefer.
    """
    mesh_white = trimesh.Trimesh(white_verts, faces, process=False)
    mesh_pial = trimesh.Trimesh(pial_verts, faces, process=False)
    # Faces whose normal has rotated past 90 degrees between white and pial.
    # NOT an inverted-face count: measured on bert, no pial from any pipeline
    # (ours, DL+DiReCT's fsr, FreeSurfer's) contains a genuinely inverted face
    # -- all are winding-consistent, positive-volume, with zero degenerate
    # triangles. This is a real ribbon-geometry signal, but it carries a mild
    # depth bias: reversed faces sit 1.4-3.3mm deeper than average, so the
    # metric slightly penalises a pial for entering sulci. The bias is ~20%
    # while pipelines differ by 60x, so rankings survive it. Nearly disjoint
    # from self-intersections (1.1% overlap), which are far more depth-biased.
    reversed_faces = (np.einsum('ij,ij->i', mesh_white.face_normals, mesh_pial.face_normals) < 0)

    fr = np.linspace(0, 1, n_samples)
    at = lambda p: seg[tuple(np.clip(np.rint(tovox(p)).astype(int), 0,
                                      np.array(seg.shape) - 1).T)]
    line = np.stack([at(white_verts + (pial_verts - white_verts) * x) for x in fr], axis=1)
    crossed = np.zeros(len(white_verts), bool)
    transit = np.zeros(len(white_verts), bool)
    for i in range(len(white_verts)):
        s = line[i]
        bg = np.where(s == 0)[0]
        if len(bg) and (s[bg[0]:] > 0).any():
            crossed[i] = True
        ix = np.where(s != 3)[0]
        if len(ix):
            after = (s[ix[0]:] == 3)
            if after.any():
                j = np.where(after)[0][0] + ix[0]
                if (s[j:] != 3).any():
                    transit[i] = True

    # TRUE face-face self-intersections. The previous metric counted vertex
    # pairs closer than 0.15mm, which is not the same thing and is biased in a
    # way that inverts the ranking: a surface packed deep into sulci has many
    # close vertex pairs by construction. Measured on bert, that proxy ranked
    # FreeSurfer's pial WORST of four (390 per 10k vertices) when a real test
    # finds it has exactly 0 intersecting faces of 271154 -- it is
    # intersection-free by design. Reported as a count of faces, so it is
    # comparable only between meshes of similar size; `self_intersect_pct`
    # alongside it is not.
    n_selfint, selfint_pct = _count_self_intersections(pial_verts, faces)
    ends_wm = (line[:, -1] == 3)
    keep = np.ones(len(white_verts), bool) if no_push is None else ~no_push
    return dict(
        end_in_wm_pinned=int((ends_wm & ~keep).sum()),
        transit_count=int(transit.sum()),
        transit_pct=100 * float(transit.mean()),
        crossed_csf_count=int(crossed.sum()),
        end_in_wm_count=int((ends_wm & keep).sum()),
        normal_reversal_pct=100 * float(reversed_faces.mean()),
        flipped_face_pct=100 * float(reversed_faces.mean()),  # deprecated alias
        self_intersections=n_selfint,
        self_intersect_pct=selfint_pct,
        mean_displacement_mm=float(np.linalg.norm(pial_verts - white_verts, axis=1).mean()),
    )


# ---------------------------------------------------------------------------
# Configurable pipeline — a single knob set spanning naive .. best, so the
# five fixes can be searched over instead of only toggled as a bundle.
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PipelineConfig:
    """Every free parameter in the field-propagated pial pipeline. The
    defaults ARE the tested-best configuration from the module docstring;
    NAIVE_CONFIG below is the other validated endpoint. Anything in
    between, or outside both, is unvalidated territory for
    field_pial_optimize.py to search."""
    use_normal_gate: bool = True
    gate_sigma_scale: float = 2.0
    gate_coherence_threshold: float = 0.7
    gate_mode: str = 'soft'
    # 1.0 matches the shipped solver, keeping --write-thickness comparable to
    # stock DiReCT; 0.35 was the surface-tested best. See module docstring (2).
    smoothing_sigma: float = 1.0
    gradient_sigma: float = None  # None = follow smoothing_sigma; see solve_velocity_field
    use_sulcal_sheet: bool = True
    dip_threshold: float = 0.95
    sheet_frac: float = 0.0
    # False: propagate from ?h.white directly. mode='best' samples the field
    # just outside WM for in-WM vertices, which covers fix #4's premise -- see
    # the module docstring note under (4).
    use_constrained_start: bool = False
    constrained_floor: float = -0.5
    # 0.5mm of clearance outside WM from round 2 on. Measured at sigma=0.35:
    # non-exempt end-in-WM 142/174 -> 12/12 and vertices under 0.5mm clearance in
    # real cortex 250/396 -> 0/3, for ~20% more flipped faces. Vertices with no
    # cortex to move into are exempt (exclude_no_cortex), which is what keeps the
    # constraint off hippocampus, midline and deep-WM closures -- unmasked it
    # would act almost entirely where it is wrong. See propagate().
    constrained_floor_after: float = 0.5
    constrained_floor_after_round: int = 1  # 0-based round at which floor_after starts
    exclude_no_cortex: bool = True  # exempt hippocampus/midline/deep-WM; see build_no_push_mask
    # Which no-cortex vertices to hold at the white surface; see build_pin_mask.
    # 'medial-wall' by default: it takes almost all of the end-in-WM benefit
    # (free-vertex end-in-WM 594/784 -> 59/60) for almost none of the folding
    # cost (flipped_face_pct 0.054/0.049 -> 0.056/0.052, against 0.157/0.147 for
    # 'all'), and leaves hippocampus/amygdala at unpinned smoothness (0.0724 vs
    # 0.0718 unpinned; 'all' gives 0.0856). The cost lands on the corpus
    # callosum instead (0.0709 -> 0.0825), where there is no cortex to damage.
    pin_scope: str = 'medial-wall'
    pin_feather: int = 2  # rings over which the pin ramps to 0 (0 = hard pin)
    # 0 = pin only vertices sitting ON a no-cortex structure. Growing the region
    # outward buys a handful of end-in-WM vertices and costs folding plus real
    # over-constraint of medial temporal cortex -- see build_no_push_mask.
    pin_rings: int = 0
    escape_in_no_cortex: bool = False  # allow the escape substitution there
    constrained_iters: int = 8
    constrained_lam: float = 0.51  # matched to pymeshlab's own filter — see build_constrained_white
    propagation_mode: str = 'best'  # 'naive' or 'best'
    propagation_rounds: int = 10
    num_integration_points: int = None  # None = solver default (10); ties to propagation_rounds
    gradient_gate: float = None  # None = solver default 1e-3; absolute, so it interacts with sigma
    step_scale: float = None  # None = full field step per round; see propagate()
    # OFF by default since the frame fix. Escape substitutes an outward step for a
    # vertex whose field step points into WM; with the field and the surface finally
    # sharing one WM boundary (--wm-from-surface) that case has largely stopped
    # arising. Measured on sub-POBHC0002_TI900TR1700_run1: turning it off changes
    # end_in_wm -- the metric it exists to improve -- by 4 vertices in lh and 0 in rh,
    # and moves 0.55% of vertices at all (whole-surface mean 0.0003 mm). Its earlier
    # validation was measured through the half-voxel frame error.
    use_escape: bool = False  # mode='best' only; see propagate()


NAIVE_CONFIG = PipelineConfig(
    use_normal_gate=False, smoothing_sigma=1.0, use_sulcal_sheet=False,
    use_constrained_start=False, propagation_mode='naive')
# The defaults above are the tested-best settings EXCEPT smoothing_sigma, which
# is 1.0 (the shipped solver's value) rather than the surface-tested 0.35 -- see
# the note in module docstring (2).
BEST_CONFIG = PipelineConfig()


def run_pipeline(config, seg, gmT, wmT, gm_prob, wm_prob, ref_img, tovox, totkr,
                  hemi_surfaces, velocity_prefix, out_dir=None, tag='',
                  thickness_dir=None, soft_seg=None, id_map=None, volume_info=None,
                  velocity_override=None):
    """Run one configuration across one or more hemispheres, reusing a
    single whole-brain solve (see main()'s docstring note on why).

    `hemi_surfaces`: {hemi_name: (white_verts, faces)}.
    `out_dir`/`tag`: if given, writes '<out_dir>/<hemi>.pial.<tag>' and, when
    `config.use_constrained_start`, '<out_dir>/<hemi>.white.corrected.<tag>'.

    `thickness_dir`: if given, writes this solve's own thickness map and
    segmentation there as 'T1w_thickmap.nii.gz' and 'seg.nii.gz', in DiReCT.py's
    layout, so extract_stats.py can consume them unchanged. kelly_kapowski
    returns a thickness map from every solve; the pial pipeline discarded it and
    the shipped pipeline paid for a second, separate DiReCT solve to obtain one.
    Note the thickness this yields is measured on THIS solve, i.e. with the
    normal gate, smoothing_sigma and fractional CSF sheet of `config` -- it is
    not numerically the same quantity as stock DiReCT's.

    Returns {hemi_name: metrics_dict} (see evaluate_surface).
    """
    gmT_use, wmT_use = gmT, wmT
    sheet_info = {}
    if config.use_sulcal_sheet:
        sheet_mask, n_blocked, dips = detect_sulcal_csf_sheet(
            seg, gm_prob, wm_prob, dip_threshold=config.dip_threshold)
        gmT_use, wmT_use = apply_fractional_sheet(gmT, wmT, sheet_mask, frac=config.sheet_frac)
        sheet_info = dict(sheet_blocked_rays=n_blocked, sheet_voxels=int(sheet_mask.sum()))

    if velocity_override is not None:
        # Nothing that distinguishes a propagation-time ablation (escape, the
        # floor, retraction, smoothing rounds) is an input to the solve, so the
        # two arms can share one field. Reusing it is not just cheaper: it makes
        # the arms provably identical upstream instead of relying on the solver
        # being deterministic. No thickness comes back, so --write-thickness is
        # refused alongside it in main().
        velocity, thickness = velocity_override, None
    else:
        velocity, thickness = solve_velocity_field(
            seg, gmT_use, wmT_use, ref_img, velocity_prefix,
            smoothing_sigma=config.smoothing_sigma, gradient_sigma=config.gradient_sigma,
            num_integration_points=config.num_integration_points,
            gradient_gate=config.gradient_gate,
            use_normal_gate=config.use_normal_gate,
            gate_sigma_scale=config.gate_sigma_scale,
            gate_coherence_threshold=config.gate_coherence_threshold, gate_mode=config.gate_mode)

    if thickness_dir is not None and thickness is not None:
        # seg does not depend on the solve, so it is always valid to write.
        save_like_direct(seg.astype(np.uint8),
                          os.path.join(thickness_dir, 'seg.nii.gz'), ref_img)
        if not np.isfinite(thickness).all() or not (thickness > 0).any():
            # A degenerate solve (e.g. smoothing_sigma small enough that the
            # gradient kernel collapses -- see direct_cuda's warning) must not
            # abort the run: the pial surfaces below are computed from the
            # velocity field and can still be useful when the thickness is not.
            # Leaving the file absent is what tells the caller to skip the
            # thickness stats, rather than writing a map of zeros that would
            # silently become NaN parcel means downstream.
            print("WARNING: thickness map is degenerate (no positive voxels); "
                  "not writing T1w_thickmap.nii.gz. Cortical thickness is "
                  "unavailable for this configuration -- surfaces still follow.",
                  file=sys.stderr)
        else:
            save_like_direct(thickness.astype(np.float32),
                              os.path.join(thickness_dir, 'T1w_thickmap.nii.gz'), ref_img)
            print("wrote T1w_thickmap.nii.gz and seg.nii.gz to %s (mean thickness over "
                  "non-zero voxels %.3f mm)"
                  % (thickness_dir, float(thickness[thickness > 0].mean())))

    # Vertices exempt from a tightened per-round floor (hippocampus, midline,
    # deep-WM closures). Built from the white surface once, so the exemption is
    # a fixed property of the vertex rather than of its current position.
    # Built per hemisphere inside the loop below, from the propagation start
    # rather than the raw white -- the pin holds vertices at that surface, so
    # the mask must describe it.
    want_mask = ((config.exclude_no_cortex or config.pin_scope != 'none')
                 and soft_seg is not None and id_map is not None)

    results = {}
    for hemi, (white_verts, faces) in hemi_surfaces.items():
        start = white_verts
        if config.use_constrained_start:
            start = build_constrained_white(
                white_verts, faces, seg, tovox, totkr, floor=config.constrained_floor,
                iters=config.constrained_iters, lam=config.constrained_lam)
            if out_dir:
                nib.freesurfer.io.write_geometry(
                    os.path.join(out_dir, '%s.white.corrected.%s' % (hemi, tag)),
                    start, faces, create_stamp=None, volume_info=volume_info)

        mask = None; pin = None
        if want_mask:
            mask = build_no_push_mask(start, faces, seg, soft_seg, id_map, tovox,
                                       rings=config.pin_rings)
            print("%s: %d/%d vertices (%.1f%%) with no cortex to move into (%s)"
                  % (hemi, mask.sum(), len(mask), 100 * mask.mean(),
                     'pinned (%s)' % config.pin_scope if config.pin_scope != 'none'
                     else 'exempt from the tightened floor'))
            pin = build_pin_mask(mask, start, faces, seg, soft_seg, id_map, tovox,
                                  scope=config.pin_scope, rings=config.pin_rings)
            if config.pin_scope != 'none':
                print("%s: of those, %d held at the white surface (scope=%s)"
                      % (hemi, pin.sum(), config.pin_scope))

        pial = propagate(start, faces, velocity, seg, tovox, totkr,
                          mode=config.propagation_mode, rounds=config.propagation_rounds,
                          floor=config.constrained_floor, lam=config.constrained_lam,
                          use_escape=config.use_escape,
                          floor_after=config.constrained_floor_after,
                          floor_after_round=config.constrained_floor_after_round,
                          no_push=mask,
                          pin_mask=pin, step_scale=config.step_scale,
                          pin_feather=config.pin_feather,
                          escape_in_no_push=config.escape_in_no_cortex)
        if out_dir:
            nib.freesurfer.io.write_geometry(
                os.path.join(out_dir, '%s.pial.%s' % (hemi, tag)),
                pial, faces, create_stamp=None, volume_info=volume_info)

        metrics = evaluate_surface(start, pial, faces, seg, tovox,
                                    no_push=pin if (pin is not None and pin.any()) else None)
        metrics.update(sheet_info)
        results[hemi] = metrics
    return results


def config_loss(metrics, n_vertices):
    """A proxy self-consistency loss — lower is better. Combines the rates
    that were treated as unambiguously bad throughout development (transit,
    folding, self-intersection, ending in WM). Deliberately excludes
    crossed_csf_count (ambiguous — can reflect a surface newly reaching CSF
    it previously couldn't, not only a genuine wrong-bank crossing) and
    mean_displacement_mm (a magnitude, not a defect rate). This is a
    starting point for field_pial_optimize.py, not a validated metric —
    redefine it for your own use rather than trusting it blindly."""
    return (metrics['transit_pct']
            + metrics.get('normal_reversal_pct', metrics['flipped_face_pct'])
            + metrics.get('self_intersect_pct', 100.0 * metrics['self_intersections'] / n_vertices)
            + 100.0 * metrics['end_in_wm_count'] / n_vertices)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--prep-dir', required=True,
                    help='directory containing seg_<Label>.nii.gz per-label logits (DiReCT.py input)')
    p.add_argument('--surf-dir', required=True,
                    help='directory containing lh.white and/or rh.white (FreeSurfer .white format) — '
                         'e.g. the pipeline output\'s surf/ directory')
    p.add_argument('--hemi', nargs='+', default=['lh', 'rh'], choices=['lh', 'rh'],
                    help='which hemisphere(s) to process (default: both)')
    p.add_argument('--out-dir', required=True, help='output directory for surfaces and logs')
    p.add_argument('--sigma', type=float, default=1.0,
                    help='smoothing_sigma for the best run: sets both the gradient '
                         '(differentiation) sigma and the hit/total accumulation sigma. '
                         'Default 1.0 matches the shipped solver so --write-thickness stays '
                         'comparable with stock DiReCT; 0.35 was the surface-tested best.')
    p.add_argument('--dip-threshold', type=float, default=0.95, help='sulcal CSF sheet dip threshold')
    p.add_argument('--write-thickness', action='store_true',
                    help="write the best solve's own thickness map and segmentation into "
                         "--prep-dir as T1w_thickmap.nii.gz / seg.nii.gz, in DiReCT.py's "
                         "layout. Lets the caller skip the separate DiReCT run (dl+direct.sh "
                         "--no-cth) that would otherwise solve the same problem a second time "
                         "just to produce them. The numbers differ from stock DiReCT's -- see "
                         "run_pipeline.")
    p.add_argument('--grad-sigma', type=float, default=None,
                    help='gradient (differentiation) sigma, overriding --sigma for that term '
                         'only. --sigma otherwise sets both the gradient sigma and the hit/total '
                         'accumulation sigma; this separates them.')
    p.add_argument('--floor-after-round', type=int, default=1,
                    help='0-based round at which --floor-after starts (default 1, i.e. after step 1)')
    p.add_argument('--constrained-start', action='store_true',
                    help="apply the constrained-white starting-surface repair (fix #4) before "
                         "propagating, and write ?h.white.corrected. Off by default: mode=best "
                         "samples the field just outside WM for in-WM vertices, which covers "
                         "fix #4's premise, and the repair was measured to leave the pial "
                         "0.038mm different while roughening the pinned population.")
    p.add_argument('--pin-rings', type=int, default=0,
                    help='mesh rings the no-cortex region grows outward from vertices sitting '
                         'on the structure itself (0 = only those vertices)')
    p.add_argument('--pin-feather', type=int, default=2,
                    help='rings over which the no-cortex pin ramps to zero (default 2; '
                         '0 = hard pin, which folds the boundary ring)')
    p.add_argument('--escape-in-no-cortex', action='store_true',
                    help='allow the escape substitution inside the no-cortex region '
                         '(off by default: there is no cortex there to escape into)')
    p.add_argument('--gradient-gate', type=float, default=None,
                    help="solver's absolute gradient-magnitude gate (default 1e-3). Lower it when "
                         "using a small --sigma, whose gradients are much smaller: 54%% of GM "
                         "voxels fall under the default gate at sigma=0.35 against 3.7%% at 1.0.")
    p.add_argument('--integration-points', type=int, default=None,
                    help="solver's num_integration_points. The saved velocity field is "
                         "per-integration-point, so the matched --rounds equals this.")
    p.add_argument('--rounds', type=int, default=10,
                    help='propagation rounds (default 10, matching the solver\'s integration points)')
    p.add_argument('--keep-total-step', action='store_true',
                    help='with --rounds, scale each step by 10/rounds so the total deformation is '
                         'unchanged -- i.e. take more, shorter steps rather than travelling further')
    p.add_argument('--no-sulcal-sheet', action='store_true',
                    help='disable the fractional sulcal-CSF-sheet repair (fix #3). Diagnostic.')
    p.add_argument('--no-normal-gate', action='store_true',
                    help='disable the interface-normal gating of velocity smoothing (fix #1) and '
                         "use the solver's plain isotropic smoothing. Diagnostic.")
    p.add_argument('--pin-scope', choices=('none', 'medial-wall', 'all'), default='medial-wall',
                    help="which no-cortex vertices to hold at the white surface (zero thickness "
                         "there). 'none': pin nothing -- they stay exempt from "
                         "--floor-after and drift ~0.14mm. 'medial-wall' (default): pin the corpus callosum "
                         "and midline closures but not hippocampus/amygdala, whose rigid block is "
                         "what roughens the surrounding pial. 'all': pin everything.")
    p.add_argument('--no-exclude', action='store_true',
                    help='do NOT exempt hippocampus/amygdala/corpus-callosum/deep-WM vertices '
                         'from --floor-after (diagnostic; the exemption is on by default)')
    p.add_argument('--floor-after', type=float, default=0.5,
                    help='signed-distance floor for the per-round relaxation from round 2 '
                         'onward (round 1 keeps --floor). 0.0 forbids the pial from sitting '
                         'inside WM at all after the first round.')
    p.add_argument('--escape', action='store_true',
                    help="re-enable the escape mechanism, which substitutes an outward step for "
                         "a vertex whose field step points into WM. OFF by default since the "
                         "frame fix: with --wm-from-surface it changes end_in_wm by 4 vertices "
                         "(lh) and 0 (rh) and moves 0.55%% of the surface. Its original "
                         "validation was measured through the half-voxel frame error.")
    p.add_argument('--no-escape', action='store_true',
                    help="accepted and ignored; escape is now off by default.")
    p.add_argument('--velocity-from', metavar='FILE',
                    help="reuse a solved velocity field (a *Velocity.nii.gz written by an "
                         "earlier run) instead of solving again. Nothing that distinguishes a "
                         "propagation-time ablation is an input to the solve, so both arms can "
                         "and should share one field -- it halves the cost and makes them "
                         "provably identical upstream. Incompatible with --write-thickness, "
                         "which needs the solve's own output.")
    p.add_argument('--wm-pv', type=int, default=3, metavar='N',
                    help="rasterize the white surfaces at partial-volume occupancy on an N-times "
                         "supersampled grid. Default 3; 0 or 1 gives the crisp staircase. Labels "
                         "always come from the crisp mask, so what this changes is only that the "
                         "solve's WM/GM priors carry the sub-voxel boundary fraction, with total "
                         "tissue conserved.")
    p.add_argument('--seg-wm', action='store_true',
                    help="take white matter for the solve, the floor and the escape rule from the "
                         "segmentation (seg == 3 on the WM logits) instead of from the white "
                         "surface's own interior. The surface is the default: it makes the field "
                         "and the surface riding it share one WM boundary rather than two that "
                         "disagree. See --wm-pv for how finely that boundary is resolved.")
    p.add_argument('--wm-from-surface', action='store_true',
                    help="accepted and ignored; this is now the default (see --seg-wm).")
    p.add_argument('--skip-naive', action='store_true',
                    help='skip the naive baseline and produce only the best-configuration pial. '
                         'The two configurations do not share a velocity field (they differ in '
                         'smoothing_sigma, the gate and the CSF sheet, all of which enter the solve), '
                         'so the baseline costs a second full GPU solve. Use this for production runs; '
                         'leave it off to reproduce the naive-vs-best comparison.')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # The segmentation, the sulcal CSF sheet, and both velocity fields are
    # whole-brain quantities — solved ONCE and reused for every hemisphere,
    # rather than repeating the (expensive) GPU solve per hemisphere.
    gm_prob, wm_prob, ref_img = load_gm_wm_probability(args.prep_dir)
    # Parcellation for the no-push exemption. Same cropped grid as seg, so no
    # resampling; absent files simply disable the exemption.
    soft_seg = id_map = None
    _soft = os.path.join(args.prep_dir, 'softmax_seg.nii.gz')
    _ldef = os.path.join(args.prep_dir, 'label_def.csv')
    if os.path.exists(_soft) and os.path.exists(_ldef):
        soft_seg = np.asarray(nib.load(_soft).dataobj)
        _df = pd.read_csv(_ldef)
        id_map = {r.LABEL: int(r.ID) for _, r in _df.iterrows()}
    elif not args.no_exclude:
        print("WARNING: softmax_seg.nii.gz / label_def.csv not found in --prep-dir; "
              "the no-cortex exemption for --floor-after is disabled.", file=sys.stderr)
    velocity_override = None
    if args.velocity_from:
        if args.write_thickness:
            sys.exit("--velocity-from cannot be combined with --write-thickness: the thickness "
                     "map is an output of the solve this option skips.")
        velocity_override = np.asarray(nib.load(args.velocity_from).dataobj)
        print("velocity: reusing %s %s (no solve)" % (args.velocity_from, velocity_override.shape))

    seg, gmT, wmT = build_seg_maps(gm_prob, wm_prob)
    tovox, totkr = make_transforms(ref_img)

    hemi_surfaces = {}
    for hemi in args.hemi:
        white_verts, faces, vinfo = nib.freesurfer.io.read_geometry(
            os.path.join(args.surf_dir, '%s.white' % hemi), read_metadata=True)
        hemi_surfaces[hemi] = (white_verts, faces)
        # A surface that states its grid lets us refuse the frame mismatch the
        # module docstring describes, instead of detecting it (weakly) after the
        # fact through check_frame_alignment. `volume` is the grid's dims; a
        # surface built on the 256^3 conform says [256,256,256] here and the
        # cropped reference says e.g. [131,141,167].
        if vinfo and 'volume' in vinfo:
            if tuple(int(x) for x in vinfo['volume']) != tuple(ref_img.shape[:3]):
                sys.exit("%s.white was built on a %s grid but the reference (seg_<Label>.nii.gz) "
                         "grid is %s: their tkrRAS frames differ by half a voxel per odd axis. "
                         "Rebuild the surfaces on the reference grid (preparedata.py --space "
                         "cropped) rather than propagating from a misplaced surface."
                         % (hemi, list(int(x) for x in vinfo['volume']), list(ref_img.shape[:3])))
        else:
            print("WARNING: %s.white carries no volume geometry, so its grid cannot be checked "
                  "against the reference; only the (coarse) frame check below guards it."
                  % hemi, file=sys.stderr)
        dist, frac_wm, in_bounds = check_frame_alignment(white_verts, seg, tovox)
        print("frame check %s: mean |distance to WM boundary| %.3f mm "
              "(expect <1.5; %.1f%% sample WM, %.1f%% in bounds)"
              % (hemi, dist, 100 * frac_wm, 100 * in_bounds))
        if dist > 1.5 or in_bounds < 0.99:
            print("WARNING: frame alignment looks wrong for %s (see module docstring) — "
                  "results below are not trustworthy until this is fixed." % hemi, file=sys.stderr)

    if not args.seg_wm:
        _m = (np.zeros(tuple(ref_img.shape[:3]), np.float32) if args.wm_pv > 1
              else np.zeros(tuple(ref_img.shape[:3]), bool))
        _crisp = np.zeros(tuple(ref_img.shape[:3]), bool)
        for _h in ('lh', 'rh'):
            _p = os.path.join(args.surf_dir, '%s.white' % _h)
            if not os.path.exists(_p):
                sys.exit("--wm-from-surface needs both white surfaces; %s is missing." % _p)
            _v, _f = (hemi_surfaces[_h] if _h in hemi_surfaces
                      else nib.freesurfer.io.read_geometry(_p))
            _crisp |= rasterize_mesh(tovox(_v), _f, tuple(ref_img.shape[:3]))
            if args.wm_pv > 1:
                _m = _m + rasterize_mesh_pv(tovox(_v), _f, tuple(ref_img.shape[:3]), args.wm_pv)
            else:
                _m = _crisp
        if args.wm_pv > 1:
            _m = np.clip(_m, 0.0, 1.0)
            print("WM mask: partial volume, supersample %d, %d partial voxels"
                  % (args.wm_pv, int(((_m > 0) & (_m < 1)).sum())))
        _before = int((seg == 3).sum())
        seg, gmT, wmT, _dem, _pro = reconcile_seg_with_surface(
            seg, gmT, wmT, _m, label_mask=(_crisp if args.wm_pv > 1 else None))
        print("WM from surface: %d -> %d voxels (%d demoted to cortex, %d promoted from "
              "non-WM interior)" % (_before, int((seg == 3).sum()), _dem, _pro))

    naive_config = dataclasses.replace(NAIVE_CONFIG)
    best_config = dataclasses.replace(BEST_CONFIG, smoothing_sigma=args.sigma, dip_threshold=args.dip_threshold,
                                       use_escape=args.escape,
                                       gradient_sigma=args.grad_sigma,
                                       propagation_rounds=args.rounds,
                                       num_integration_points=args.integration_points,
                                       gradient_gate=args.gradient_gate,
                                       step_scale=(10.0/args.rounds) if args.keep_total_step else None,
                                       constrained_floor_after=args.floor_after,
                                       constrained_floor_after_round=args.floor_after_round,
                                       exclude_no_cortex=not args.no_exclude,
                                       pin_scope=args.pin_scope,
                                       use_normal_gate=not args.no_normal_gate,
                                       use_sulcal_sheet=not args.no_sulcal_sheet,
                                       pin_feather=args.pin_feather,
                                       pin_rings=args.pin_rings,
                                       use_constrained_start=args.constrained_start,
                                       escape_in_no_cortex=args.escape_in_no_cortex)

    if args.skip_naive:
        print("\n=== naive baseline skipped (--skip-naive) ===")
    else:
        print("\n=== naive (whole-brain solve) ===")
        naive_results = run_pipeline(naive_config, seg, gmT, wmT, gm_prob, wm_prob, ref_img, tovox, totkr,
                                      hemi_surfaces, os.path.join(args.out_dir, 'naive_'),
                                      out_dir=args.out_dir, tag='field_naive',
                                      volume_info=volume_info_from_prep(args.prep_dir))
        for hemi, metrics in naive_results.items():
            print("--- %s naive ---" % hemi)
            print(metrics)

    print("\n=== best (whole-brain solve) ===")
    best_results = run_pipeline(best_config, seg, gmT, wmT, gm_prob, wm_prob, ref_img, tovox, totkr,
                                 hemi_surfaces, os.path.join(args.out_dir, 'best_'),
                                 out_dir=args.out_dir, tag='field_best',
                                 thickness_dir=args.prep_dir if args.write_thickness else None,
                                 soft_seg=soft_seg, id_map=id_map,
                                 volume_info=volume_info_from_prep(args.prep_dir),
                                 velocity_override=velocity_override)
    for hemi, metrics in best_results.items():
        print("--- %s best ---" % hemi)
        print(metrics)


if __name__ == '__main__':
    main()
