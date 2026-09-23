#!/usr/bin/env python
"""Build DiReCT's inputs from SURFACES rather than from the label volume.

Both boundaries are meshed and then rasterized as partial volume onto the
solve's grid:

    wmT = pv(WM surface)
    gmT = pv(GM surface) - pv(WM surface)        the ribbon's occupancy
    seg = 3 where pv(WM) > 0.5, 2 where pv(GM) > 0.5, else 0

Two things this buys.

SUB-THRESHOLD CSF. The GM surface is the isosurface of the cortical ribbon
AFTER a topology correction. The ribbon carries hundreds of handles because
sulcal banks touch wherever the CSF between them fell below detection; the
voxels the confidence-ordered growth declines to add are exactly those bridges
-- tissue by the argmax, CSF by the topology. Meshing the corrected ribbon
therefore hands the solve a GM boundary with those sulci OPEN. Measured on one
hemisphere: 4200 of 498792 voxels (0.84%), and crossed_csf on the resulting
pial fell 594 -> 336.

ONE FRAME CONVERSION. The pipeline's usual route reconciles a voxel
segmentation against the white surface by rasterising it, which on the measured
subject promoted 71550 voxels -- the two representations disagree and one is
forced onto the other. Here both boundaries are surfaces from the start, the
conformed -> cropped mapping is applied once to both, and there is nothing left
to reconcile.

The label volume (mri/aparc.atlas+aseg.nii.gz) is usually on the 256^3 conform
while the solve runs on the cropped grid. That mapping is done here, in one
place, rather than in each caller.

STATUS: NOT A DEFAULT, and measured worse than the logits route on every
containment metric. Kept because the sub-threshold CSF idea is worth another
attempt, not because this implementation of it works.

What the measurements showed, in the order they were established:

  1. The reported benefit was a SCORING ARTEFACT. crossed_csf was computed
     against each arm's OWN segmentation. This arm's segmentation has the sulci
     opened, so a vertex that stops inside a sulcus is in background and stays
     there -- no crossing is recorded -- while the same vertex against the
     baseline segmentation goes tissue/background/tissue and counts. Scored
     against a COMMON segmentation the direction reverses: 712 -> 1965, not
     703 -> 387. Any comparison of crossed_csf, transit_pct or end_in_wm
     across arms with different seg is invalid.

  2. The topology correction contributes almost none of the difference.
     correct_ribbon=False changes c_crossed by a few percent (1965 vs 1826).

  3. The partial volume contributes little either: gm_crisp=True reproduces
     91% of the effect (1796 vs 1965).

  4. It is the SMOOTHED GM SURFACE itself, and nsmooth -- inherited from the
     white surface, never tested on a GM envelope -- is the live variable.
     Sweeping it, scored against a common segmentation, 4 subjects:

       nsmooth   euc_med  euc_p95  c_crossed  c_transit  c_endwm  selfint
       logits     0.5399   1.3817      712.5     0.4358    898.0     10.0
       0          0.5143   1.3672      919.9     0.4754   1331.6     18.9
       10         0.5266   1.3434     1507.9     0.4942   1230.9     45.8
       50         0.5427   1.3443     1964.5     0.5077    898.6     77.5

     Smoothing trades containment (crossed, transit, self-intersections all
     rise) against inward excursion (end_in_wm and inward_pct fall). No value
     beats the logits baseline on containment.

  5. The detector is not specific to CSF. It finds where the ribbon is
     TOPOLOGICALLY wrong and declines to complete the loop; whether that place
     is sulcal CSF is an assumption. 89% of the cut voxels are adjacent to
     existing CSF (median distance 1.0mm vs 1.73mm for GM generally), but 12%
     are within 1mm of white matter and 38 sit in the ribbon interior touching
     neither. A handle can run anywhere.

To make the idea work, the missing piece is a LOCALITY CONSTRAINT: only accept
a cut adjacent to existing CSF, or within ~1mm of it. That discards exactly the
population that cannot be sulcal CSF, at the cost of leaving some handles
unfixed -- which does not matter, since the GM surface tolerates defects.
Untested.
"""

import os

import numpy as np
import nibabel as nib

from . import outer_surface as osf
from . import wm_labels
from . import wm_surface
from .field_pial_prototype import (get_vox2ras_tkr, load_gm_wm_probability,
                                   make_transforms, rasterize_mesh, rasterize_mesh_pv)
from .retarget_surface import tkr_to_tkr
from .surface_frames import check_surface_frame
from .topology_gpu import correct_topology

SUPERSAMPLE = 3            # partial-volume rasterisation, as the white reconciliation uses
N_BANDS = 8                # priority bands for the ribbon correction
PROTECT_ABOVE = 0.7        # never sacrifice tissue the model is this sure about
                           # (a probability: load_gm_wm_probability applies expit)
GUARD_WM_MM = 1.0          # protect tissue within this of the WM fill
GUARD_ENV_MM = 2.0         # ... and this of the closing envelope's edge
PAD = 2


def locality_guard(seg_labelled, df_labels, region, excluded, ribbon,
                   wm_mm=GUARD_WM_MM, env_mm=GUARD_ENV_MM):
    """Tissue the topology correction may NOT sacrifice, by location.

    The correction has no notion of CSF -- it finds where the ribbon's shape is
    topologically wrong, and a handle can run anywhere. Two zones are therefore
    put out of bounds:

      * within `wm_mm` of the WM FILL. Note the fill, not the
        Left/Right-Cerebral-White-Matter label: the fill also contains the
        thalamus, caudate, putamen, pallidum and ventricles, whose interiors
        are several mm from any cerebral-WM voxel. Guarding on the label alone
        left 49 rh cuts inside the pallidum and putamen, up to 5.8mm from
        cerebral WM -- and none in lh, which is why a one-hemisphere check
        missed it.
      * within `env_mm` of the closing envelope's edge, i.e. the outer brain
        margin. The envelope bridges sulci, so sulcal depths are inside it
        while the outer cortical surface is on it; plain distance to background
        cannot separate those, because sulcal CSF connects outward. Measured,
        this one earns little (it excludes ~5% more cuts, all of them already
        CSF-adjacent) but it is cheap and principled.

    Measured at floor 0.70: the WM-fill guard takes rh from 49 cuts inside the
    fill to 0, deep cuts (>2mm from any CSF) 86 -> 41, and CSF-adjacency
    85.9% -> 87.7%, for 49 of 2143 cuts.
    """
    from scipy.ndimage import distance_transform_edt, binary_fill_holes
    from . import outer_surface as _osf
    side = 'Left' if region == 'lh' else 'Right'
    fill = np.isin(np.asarray(seg_labelled),
                   wm_labels.hemisphere_labels(df_labels, side, excluded))
    guard = distance_transform_edt(~fill) <= wm_mm
    if env_mm:
        env = binary_fill_holes(_osf.close_by_ball(ribbon, 6.0))
        guard = guard | (distance_transform_edt(env) <= env_mm)
    return guard


def hemisphere_ribbon(seg_labelled, df_labels, region, excluded):
    """The cortical ribbon for one hemisphere: WM fill plus the cortex parcels.

    The WM fill's labels are 'Left-*' / 'Right-*'; a parcellated aseg names the
    cortex per gyrus as 'lh-*' / 'rh-*'. Both are needed -- without the parcels
    this is the WM mask, not the ribbon.
    """
    return np.isin(np.asarray(seg_labelled),
                   wm_labels.ribbon_labels(df_labels, region, excluded))


def tissue_priority(gm_prob, wm_prob, ref_img, label_img):
    """Per-voxel 'how sure is the model this is tissue', on the LABEL grid.

    max(P_wm, P_ctx), a PROBABILITY in [0, 1]: load_gm_wm_probability already
    applies expit() to the per-label logits, so these are sigmoid outputs, not
    logits. On the cortical ribbon the distribution runs median 0.866, p25
    0.707, p10 0.543, p01 0.080.

    High in the interior of either tissue, low at the CSF boundary where both
    are weak. NOT P(WM|WM,cortex) -- that asks WHICH tissue, so on a GM+WM mask
    it ranks confident cortex as low confidence and the growth cuts straight
    through the ribbon (measured: cuts at the ribbon's own median confidence,
    i.e. no selectivity at all).

    Outside the reference image's extent the priority is set to the maximum, so
    those voxels are annexed first rather than sacrificed.
    """
    raw = np.maximum(np.asarray(wm_prob), np.asarray(gm_prob)).astype(np.float32)
    hi = float(raw.max())
    sh = tuple(label_img.shape[:3])
    if sh == tuple(ref_img.shape[:3]) and np.allclose(label_img.affine, ref_img.affine):
        return raw, hi
    from scipy.ndimage import map_coordinates
    g = np.meshgrid(*[np.arange(k) for k in sh], indexing='ij')
    idx = np.stack([g[0].ravel(), g[1].ravel(), g[2].ravel(), np.ones(g[0].size)])
    T = np.linalg.inv(ref_img.affine) @ label_img.affine    # label vox -> ref vox
    cc = (T @ idx)[:3]
    out = map_coordinates(raw, cc, order=1, mode='constant', cval=hi).reshape(sh)
    inb = (cc >= 0).all(0) & (cc <= (np.array(raw.shape) - 1)[:, None]).all(0)
    return np.where(inb.reshape(sh), out, hi).astype(np.float32), hi


FRAME_MARGIN_PCT = 2.0     # see _assert_solve_frame; the real gap is ~20 points


def _assert_solve_frame(hemi, verts, M, label_img):
    """Reject wm_surfaces= handed over in the LABEL grid's tkrRAS frame.

    The parameter takes surfaces "already in the solve's frame" on a comment
    alone. Label-grid surfaces are accepted silently and shift the whole mesh
    by whatever the crop moved the tkr origin -- 7.0mm on OAS30001 -- which
    surfaced only as 18919 self-intersections against a correct 5905.

    The test is this repo's established one: surface_frames.check_surface_frame
    sampling tissue labels at the vertices, which its docstring argues is the
    conclusive check where centroid, bounding-box and nearest-neighbour tests
    are not. The shape check in pial_clean.prepare is not an option here --
    it reads volume_info off the file, and in-memory surfaces carry none.

    But check_surface_frame's ABSOLUTE threshold is not sufficient for this.
    It is deliberately blind to small displacements, and measured on
    OAS30001's white surfaces the wrong reading still samples 75.6% (lh) /
    76.3% (rh) tissue -- above the 72% pass mark, so it would be waved
    through -- against 95.7% / 95.8% for the right one. So the check is used
    COMPARATIVELY instead: score the mesh read as solve-frame (M^-1 v) against
    the same mesh read as label-frame (v), and reject when the label-frame
    reading is the better one. That has no fixed floor, which matters because
    the offset between the frames is set by the ribbon's bounding box and can
    be smaller on another subject; and it cannot cry wolf when the two frames
    coincide, since M == I makes the two readings identical.

    Everything stays inside the label grid, so no scanner-RAS affine enters.
    """
    W = label_img.affine @ np.linalg.inv(get_vox2ras_tkr(label_img))
    Minv = np.linalg.inv(M)
    verts = np.asarray(verts, float)
    as_solve = (Minv[:3, :3] @ verts.T).T + Minv[:3, 3]
    _, st_solve = check_surface_frame(as_solve, W, label_img, kind='white')
    _, st_label = check_surface_frame(verts, W, label_img, kind='white')
    if st_label['tissue'] > st_solve['tissue'] + FRAME_MARGIN_PCT:
        raise ValueError(
            "wm_surfaces[%r] is not in the solve grid's tkrRAS frame: its "
            "vertices sample %.1f%% tissue read as LABEL-grid coordinates "
            "against %.1f%% read as solve-grid ones, so they are in the label "
            "grid's frame. The likely cause is a surface from "
            "wm_surface.build_hemisphere, which returns vertices in the (here "
            "cropped) label grid's tkrRAS, passed on unmapped. Fix by applying "
            "retarget_surface.tkr_to_tkr(prep_dir, ref_img, src_ref=label_img) "
            "to the vertices first -- the same mapping this function applies "
            "to the surfaces it builds itself."
            % (hemi, st_label['tissue'], st_solve['tissue']))


def build_surface_segmentation(prep_dir, hemis=('lh', 'rh'), nsmooth=None,
                               n_bands=N_BANDS, supersample=SUPERSAMPLE,
                               correct_ribbon=True, gm_crisp=False,
                               protect_above=PROTECT_ABOVE, wm_surfaces=None,
                               locality=True, crop=True, topology='nighres',
                               verbose=True):
    """Surfaces -> (seg, gmT, wmT) on the solve's grid, plus the WM surfaces.

    Returns a dict with seg/gmT/wmT/ref_img/tovox/totkr, 'surfaces' (the WM
    surfaces, in the cropped tkrRAS frame, ready to propagate), 'gm_surfaces',
    and 'found_csf' (voxels per hemisphere the correction called CSF).

    correct_ribbon=False meshes the raw ribbon instead, i.e. skips the
    sub-threshold CSF detection -- the control for it.

    protect_above is a confidence floor on the ribbon correction: tissue at or
    above it is never sacrificed, so the sulcus is only opened where the model
    was unsure. The topology is then not guaranteed genus 0 -- which does not
    matter here, since the GM surface is allowed defects. In the ribbon's units
    (max of the WM and cortex logits) the ribbon median is ~0.87 and the voxels
    the unprotected correction removes have median ~0.65.

    wm_surfaces optionally supplies {hemi: (verts, faces)} to use as the WM
    boundary instead of building one. Pass the surface you intend to PROPAGATE
    from: otherwise wmT is rasterised from a different mesh than the one the
    propagation starts on, and the field is shaped around a boundary the
    starting surface does not sit on. That mismatch invalidated a
    36-hemisphere comparison -- it moves exactly the metrics a displaced
    starting mesh would move (self-intersections, slide, inward). Surfaces
    passed here must be in the SOLVE's tkrRAS, not the label grid's; that is
    checked, see _assert_solve_frame.

    topology selects the correction for the WHITE surface built here, and is
    what pial_pipeline's --topology now reaches: this function used to hardcode
    'gpu' at the build_hemisphere call, so the flag was silently inert on the
    default `surface-pv` path.

    It deliberately does NOT govern the GM ribbon's correct_topology call
    below. Removal-only is the right behaviour for the GM envelope -- the
    sub-threshold CSF this module exists for is found by declining to ADD the
    bridging voxels -- and the ribbon is the only caller that uses
    protect_above/protect_mask, which the GPU implementation provides and
    nighres does not.

    crop=True (the default) builds on the ribbon's bounding box rather than on
    whatever grid the label volume was written on -- 4.9x fewer voxels than a
    256^3 conform, for the same surfaces to a p95 of 0.0002mm. See
    wm_surface.load_inputs.

    gm_crisp=True takes the GM occupancy as the voxel-centre-inside test rather
    than the partial-volume fraction. The PV fraction gives gmT a soft outer
    edge, so DiReCT's speed term is still non-zero half a voxel beyond the GM
    surface and the flow can push past it; the crisp mask stops exactly at the
    surface. WM stays partial-volume either way, so the change is isolated to
    the outer boundary.
    """
    nsmooth = wm_surface.NSMOOTH_DEFAULT if nsmooth is None else nsmooth
    gm, wm, ref_img = load_gm_wm_probability(prep_dir)
    tovox, totkr = make_transforms(ref_img)
    shape = tuple(ref_img.shape[:3])
    seg_lab, df, aff_lab, excluded, label_img = wm_surface.load_inputs(prep_dir, crop=crop)
    if verbose and crop:
        print('label volume cropped to %s (%.2fM voxels)'
              % (tuple(seg_lab.shape), seg_lab.size / 1e6))
    prio, hi = tissue_priority(gm, wm, ref_img, label_img)
    # the label grid's tkrRAS -> the solve's tkrRAS, applied once, to both surfaces
    M = tkr_to_tkr(prep_dir, ref_img, src_ref=label_img)
    to_ref = lambda v: (M[:3, :3] @ np.asarray(v).T).T + M[:3, 3]

    pv_wm = np.zeros(shape, np.float32)
    pv_gm = np.zeros(shape, np.float32)
    wm_out, gm_surfaces, found = {}, {}, {}   # NOT wm_surfaces: that is the parameter
    for h in ('lh', 'rh'):
        ribbon = hemisphere_ribbon(seg_lab, df, h, excluded)
        if correct_ribbon:
            pmask = (locality_guard(seg_lab, df, h, excluded, ribbon)
                     if locality else None)
            corr, info = correct_topology(
                np.pad(ribbon, PAD), verbose=False, pair='26-6',
                priority=np.pad(prio, PAD, constant_values=hi), n_bands=n_bands,
                protect_above=protect_above,
                protect_mask=(None if pmask is None else np.pad(pmask, PAD)))
            corr = corr[PAD:-PAD, PAD:-PAD, PAD:-PAD]
            found[h] = int((ribbon & ~corr).sum())
            if verbose:
                print('%s ribbon: %d voxels found as sub-threshold CSF (%.2f%%)'
                      % (h, found[h], 100 * found[h] / max(ribbon.sum(), 1)))
        else:
            corr, found[h] = ribbon, 0
        gv, gf = osf.mesh_envelope(corr, aff_lab, nsmooth=nsmooth)
        gv = to_ref(gv)
        if wm_surfaces is not None:
            wv, wf = wm_surfaces[h]           # must already be in the solve's frame
            wv = np.asarray(wv)
            _assert_solve_frame(h, wv, M, label_img)
        else:
            wv, wf = wm_surface.build_hemisphere(seg_lab, df, aff_lab, h, excluded,
                                                 nsmooth=nsmooth, topology=topology,
                                                 priority=prio)
            wv = to_ref(wv)
        gm_surfaces[h] = (gv, np.asarray(gf))
        wm_out[h] = (wv, np.asarray(wf))
        pv_gm += (rasterize_mesh(tovox(gv), np.asarray(gf), shape).astype(np.float32)
                  if gm_crisp else
                  rasterize_mesh_pv(tovox(gv), np.asarray(gf), shape, supersample))
        pv_wm += rasterize_mesh_pv(tovox(wv), np.asarray(wf), shape, supersample)

    pv_gm = np.clip(pv_gm, 0.0, 1.0)
    pv_wm = np.clip(pv_wm, 0.0, 1.0)
    seg = np.where(pv_wm > 0.5, 3, np.where(pv_gm > 0.5, 2, 0)).astype(np.uint8)
    gmT = np.clip(pv_gm - pv_wm, 0.0, 1.0).astype(np.float32)
    wmT = pv_wm.astype(np.float32)
    if verbose:
        print('surface-derived segmentation (%s GM): WM %d, GM %d voxels'
              % ('crisp' if gm_crisp else 'PV',
                 int((seg == 3).sum()), int((seg == 2).sum())))
    return dict(seg=seg, gmT=gmT, wmT=wmT, ref_img=ref_img, tovox=tovox, totkr=totkr,
                surfaces={h: wm_out[h] for h in hemis},
                gm_surfaces={h: gm_surfaces[h] for h in hemis},
                found_csf=found, pv_gm=pv_gm, pv_wm=pv_wm, prep_dir=prep_dir)
