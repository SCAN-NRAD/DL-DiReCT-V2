#!/usr/bin/env python
"""Solve the DiReCT velocity field and propagate a surface along it, in one call.

This is the two stages of pial_clean.py behind a single entry point, with the
surface an optional in-memory input and the propagation available on either
device:

  propagate_on='cpu'    numpy/scipy, float64, the reference implementation
  propagate_on='cuda'   torch, the field STAYS in GPU memory from the solve
                        through the propagation -- it is never copied to the
                        host unless the caller asks for it

Equivalence of the two paths
----------------------------
The CPU round is

    cur = build_constrained_white(cur + step, ..., floor=-inf, iters=k)

and with floor=-inf the signed-distance constraint can never fire, so the round
is a plain Taubin relaxation; the sdt and its gradient are computed and unused.
Two further facts let the GPU path drop every per-round affine:

  * totkr is affine, so with A the vox2ras_tkr linear block,
        step = (totkr(pos - v) - totkr(pos)) * S = A @ (-v) * S
    and applying it to a tkrRAS point is exactly `pos -= v * S` in voxel space.
  * the pin blend w*start + (1-w)*cur is an affine combination (the weights sum
    to one), so it commutes with the affine.

The GPU path therefore runs the whole loop in voxel coordinates and converts
once at the end. It is the same arithmetic, not an approximation; the residual
difference against the CPU path is float32-vs-float64 rounding. Run this module
with --check to measure it.

Sampling matches too: scipy's map_coordinates(order=1, mode='nearest') is
trilinear with the edge value held outside, which is grid_sample's
mode='bilinear', padding_mode='border', align_corners=True.
"""

import argparse
import os
import sys

import numpy as np
import nibabel as nib
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from . import pial_clean as pc
from . import solve_grid
from . import surface_seg
from . import wm_surface
from .field_pial_prototype import (_pin_weights, build_no_push_mask,
                                   build_pin_mask, evaluate_surface, get_vox2ras_tkr)
from .surface_frames import volume_info_from_image


# ---------------------------------------------------------------------------
# the CUDA propagation
# ---------------------------------------------------------------------------
def adjacency_from_faces(n_vertices, faces):
    """(W, deg): the vertex neighbour-indicator matrix straight from the faces.

    Identical structure and degrees to the trimesh-based builder in
    field_pial_prototype, but that walks vertex_neighbors in Python, which
    measured 1.30s against 0.06s here on a 134k-vertex hemisphere -- roughly 90%
    of a GPU propagation. Only the matrix is needed on this path, not the mesh.
    """
    F = np.asarray(faces)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])                 # undirected
    W = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])),
                      shape=(n_vertices, n_vertices)).tocsr()
    W.sum_duplicates()
    W.data[:] = 1.0                                     # indicator, not multiplicity
    deg = np.asarray(W.sum(1)).ravel()
    deg[deg == 0] = 1
    return W, deg


def _sparse_adjacency(Wm, device, dtype):
    """scipy CSR neighbour-indicator matrix -> torch sparse CSR on `device`."""
    Wm = Wm.tocsr()
    return torch.sparse_csr_tensor(
        torch.from_numpy(Wm.indptr.astype(np.int64)).to(device),
        torch.from_numpy(Wm.indices.astype(np.int64)).to(device),
        torch.from_numpy(Wm.data.astype(np.float64)).to(device=device, dtype=dtype),
        size=Wm.shape)


def _sample_trilinear(field, pos):
    """field [1, 3, D, H, W], pos [N, 3] in (d, h, w) voxel indices -> [N, 3].

    Equivalent to map_coordinates(order=1, mode='nearest') per component.
    """
    D, H, W = field.shape[2:]
    size = pos.new_tensor([D, H, W])
    # align_corners=True: index i maps to 2i/(n-1) - 1. grid's last axis is
    # (x, y, z) = (w, h, d), the REVERSE of the index order.
    g = 2.0 * pos / (size - 1).clamp(min=1) - 1.0
    grid = g.flip(-1).reshape(1, -1, 1, 1, 3)
    out = F.grid_sample(field, grid, mode='bilinear', padding_mode='border',
                        align_corners=True)
    return out[0, :, :, 0, 0].transpose(0, 1)


def propagate_pial_torch(white_verts, faces, velocity, tovox_affine, totkr_affine,
                         pin_mask=None, device=None, dtype=torch.float32,
                         rounds=pc.ROUNDS, step_scale=pc.STEP_SCALE,
                         relax_iters=pc.RELAX_ITERS,
                         relax_iters_final=pc.RELAX_ITERS_FINAL,
                         relax_lambda=pc.RELAX_LAMBDA, pin_feather=pc.PIN_FEATHER):
    """Carry a surface along the velocity field, entirely on the GPU.

    `velocity` is either the [1, 3, D, H, W] tensor solve_velocity_field_t
    returns (used in place, nothing is copied) or a [D, H, W, 3] array, which is
    uploaded once. `tovox_affine` / `totkr_affine` are the 4x4 matrices, not the
    callables -- the loop needs the linear block, not a host round-trip.

    Returns the propagated vertices as a float64 numpy array in tkrRAS.
    """
    if device is None:
        device = velocity.device if torch.is_tensor(velocity) else \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)

    if torch.is_tensor(velocity):
        field = velocity.to(device=device, dtype=dtype)
    else:
        field = torch.from_numpy(np.ascontiguousarray(velocity)).to(device=device, dtype=dtype)
        field = field.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    if field.ndim != 5 or field.shape[1] != 3:
        raise ValueError('velocity must be [1, 3, D, H, W] or [D, H, W, 3], got %s'
                         % (tuple(field.shape),))

    Wm, deg = adjacency_from_faces(len(np.asarray(white_verts)), faces)
    Wt = _sparse_adjacency(Wm, device, dtype)
    degt = torch.from_numpy(deg).to(device=device, dtype=dtype).unsqueeze(1)

    Ainv = torch.from_numpy(np.asarray(tovox_affine, np.float64)).to(device=device, dtype=dtype)
    A = torch.from_numpy(np.asarray(totkr_affine, np.float64)).to(device=device, dtype=dtype)
    v_tkr = torch.from_numpy(np.asarray(white_verts, np.float64)).to(device=device, dtype=dtype)
    pos = v_tkr @ Ainv[:3, :3].T + Ainv[:3, 3]          # voxel indices
    start = pos.clone()

    pin_w = None
    if pin_mask is not None and np.asarray(pin_mask).any():
        pin_w = torch.from_numpy(_pin_weights(np.asarray(pin_mask), Wm, pin_feather)) \
            .to(device=device, dtype=dtype).unsqueeze(1)

    for rnd in range(rounds):
        pos = pos - _sample_trilinear(field, pos) * step_scale
        iters = relax_iters if rnd < rounds - 1 else relax_iters_final
        for i in range(iters):
            neighbour_mean = torch.sparse.mm(Wt, pos) / degt
            mu = -0.53 if i % 2 else relax_lambda
            pos = pos + mu * (neighbour_mean - pos)
        if pin_w is not None:
            pos = pin_w * start + (1.0 - pin_w) * pos

    out = pos @ A[:3, :3].T + A[:3, 3]
    return out.double().cpu().numpy()


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------
def propagate(white_verts, faces, velocity, seg, tovox, totkr, ref_img=None,
              pin_mask=None, on='cuda', device=None, dtype=torch.float32):
    """Propagate on 'cpu' (numpy reference) or 'cuda' (field resident on GPU).

    On the CPU path `velocity` must be, or be convertible to, the [D, H, W, 3]
    array; a GPU tensor is brought across once.
    """
    on = str(on).lower()
    if on not in ('cpu', 'cuda'):
        raise ValueError("propagate_on must be 'cpu' or 'cuda', got %r" % (on,))
    if on == 'cuda' and not torch.cuda.is_available() and device is None:
        print('WARNING: propagate_on="cuda" but no CUDA device; running torch on CPU',
              file=sys.stderr)
    if on == 'cpu':
        if torch.is_tensor(velocity):
            velocity = pc.velocity_to_numpy(velocity)
        return pc.propagate_pial(white_verts, faces, velocity, seg, tovox, totkr,
                                 pin_mask=pin_mask)
    if ref_img is None:
        raise ValueError('the cuda path needs ref_img for the vox2ras_tkr affine')
    A = get_vox2ras_tkr(ref_img)
    return propagate_pial_torch(white_verts, faces, velocity, np.linalg.inv(A), A,
                                pin_mask=pin_mask, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# solve + propagate
# ---------------------------------------------------------------------------
WHITE_BUILD = 'white_build.json'


def _white_build_record(nsmooth, topology, crop, segmentation='surface-pv'):
    return dict(nsmooth=int(nsmooth), topology=str(topology), crop=bool(crop),
                segmentation=str(segmentation))


def _load_white_for_reuse(where, prep_dir, hemis, nsmooth, topology, crop, verbose):
    """{hemi: (verts, faces)} from `where`, ready for wm_surfaces=.

    `where` may be an absolute path or a directory name inside prep_dir.

    NO FRAME CONVERSION HAPPENS HERE, and none is needed: _solve_and_propagate
    writes ?h.white from `outer`, i.e. already in the solve's tkrRAS, and the
    solve grid is load_gm_wm_probability's ref_img, which is the prep's own
    on-disk grid. tkr_to_tkr(prep, ref_img) is the identity between them.
    Converting with the CROPPED label grid's matrix instead -- which is the one
    build_surface_segmentation uses internally for the surfaces it builds --
    shifts the mesh by the ribbon crop, 5.5mm in z on OAS30458, and moves every
    pial vertex by a median 5.4mm. Measured, not hypothetical: that is what the
    first version of this did.
    """
    import json
    d = where if os.path.isdir(where) else os.path.join(prep_dir, where)
    if not os.path.isdir(d):
        raise FileNotFoundError('reuse_white: no such directory %s' % d)
    rec_path = os.path.join(d, WHITE_BUILD)
    want = _white_build_record(nsmooth, topology, crop)
    if os.path.exists(rec_path):
        have = json.load(open(rec_path))
        bad = {k: (have.get(k), want[k]) for k in want if have.get(k) != want[k]}
        if bad:
            raise ValueError(
                'reuse_white: %s was built with %s, this call wants %s. Reusing it '
                'would label the output with this call\'s parameters while starting '
                'from the other surface.'
                % (d, {k: v[0] for k, v in bad.items()}, {k: v[1] for k, v in bad.items()}))
    elif verbose:
        print('WARNING: %s has no %s, so the surfaces there cannot be checked against '
              'nsmooth=%s topology=%s crop=%s. Reusing on the caller\'s word.'
              % (d, WHITE_BUILD, nsmooth, topology, crop), file=sys.stderr)
    out = {}
    for h in hemis:
        p = os.path.join(d, '%s.white' % h)
        if not os.path.exists(p):
            raise FileNotFoundError('reuse_white: %s missing' % p)
        v, f = nib.freesurfer.io.read_geometry(p)
        out[h] = (np.asarray(v, np.float64), np.asarray(f))
    return out


def reconstruct(prep_dir, surf_dir=None, surfaces=None, hemis=('lh', 'rh'),
                propagate_on='cuda', velocity=None, pin=True, out_dir=None,
                verbose=True, report=None, compute_thickness=False,
                build_white=None, nsmooth=wm_surface.NSMOOTH_DEFAULT,
                topology='nighres', segmentation='logits', crop=True,
                solve_margin=solve_grid.MARGIN,
                velocity_sigma=pc.VELOCITY_SIGMA,
                write_white=True, stats=False, subject_id=None,
                reuse_white=None, correct_ribbon=True, wm_from_surface=False,
                dtype=torch.float32, device=None):
    """Solve the field and propagate, returning the propagated surfaces.

    prep_dir        a --space cropped prep (seg_<Label>.nii.gz, softmax_seg.nii.gz,
                    label_def.csv)
    surf_dir        directory holding ?h.white; ignored if `surfaces` is given
    surfaces        optional {hemi: (verts, faces)} already in the cropped tkrRAS
                    frame. Both hemispheres are required (the WM label is
                    reconciled against their union), but only `hemis` are
                    propagated.
    build_white     build the white surfaces here from the prep's segmentation
                    (nighres topology correction + marching cubes + `nsmooth`
                    Taubin steps) instead of reading ?h.white. Defaults to True
                    when neither surf_dir nor surfaces is given, so the whole
                    chain segmentation -> white -> field -> pial runs in one
                    process with nothing going through disk.
    topology        'nighres' (default) or 'gpu' for the topology correction
                    when building the white surfaces. It applies on BOTH
                    routes: under `surface-pv` it is forwarded to
                    surface_seg.build_surface_segmentation, which used to
                    hardcode 'gpu' and so ignored this flag entirely.
    write_white     also write ?h.white beside ?h.pial when out_dir is given
                    (default). It is the mesh the propagation started from, in
                    the pial's frame; under `surface-pv` nothing else writes it.
    correct_ribbon  `surface-pv` only. True (default) runs the sub-threshold CSF
                    detection on the GM ribbon before meshing it, opening sulci
                    the model scored below detection. False meshes the raw
                    ribbon -- surface_seg documents it as "the control for it".
                    Note the `surface-pv` route turns on three things at once
                    (PV rasterisation of the WM boundary, a meshed GM envelope,
                    and this correction), so an arm that differs only in this
                    flag is what separates the third from the first two.
    reuse_white     directory holding ?h.white from an EARLIER run of this same
                    pipeline on this same prep. Under `surface-pv` the white
                    surfaces are rebuilt from the segmentation on every call --
                    topology correction, marching cubes and `nsmooth` Taubin
                    steps, 34 of the 61s the segmentation build costs. Reusing
                    them makes a second solve at a different --velocity-sigma
                    cost the solve, not the surfaces. Measured: 60.9s -> 27.0s
                    for the segmentation build, with seg bit-identical and the
                    recovered mesh 4.3e-06 mm from the rebuilt one (float32
                    storage).

                    ONLY valid when those surfaces were built with the same
                    nsmooth/topology/crop. The pipeline writes white_build.json
                    beside ?h.white recording them, and this REFUSES a mismatch;
                    surfaces from before that file existed are accepted with a
                    warning, since the alternative is refusing every surface
                    already on disk.
    stats           also write regional_stats' result-thick-<metric>.csv /
                    result-thickstd-<metric>.csv into out_dir (field, field_raw,
                    travel, nn, sym_nn). Needs out_dir and the prep's
                    aparc.atlas+aseg.nii.gz. It reuses the segmentation, the
                    white meshes, the pials and the field this call already
                    holds, so it costs the aggregation alone (~2 s) rather than
                    the ~35 s rebuild a standalone regional_stats run pays.
    subject_id      the SUBJECT cell of those CSVs; defaults to prep_dir's
                    directory name.
    solve_margin    solve and propagate on the cerebrum plus this many voxels of
                    background instead of on the whole supplied grid. The
                    supplied grid is the bounding box of the brain mask with NO
                    margin, so the tissue touches the faces and the propagation
                    has nothing to stop against there, while the cerebellum end
                    carries 24-29 slices the solve has no use for. None keeps
                    the supplied grid. Ignored when `velocity` is given, since
                    the field defines its own grid. See solve_grid.
    crop            build the surfaces on a crop of the label volume rather than
                    on whatever grid it was written on (4.9x fewer voxels than a
                    256^3 conform). See wm_surface.load_inputs; crop=False
                    reproduces the pre-crop behaviour.
    segmentation    'logits' (DEFAULT) takes seg/gmT/wmT straight from the model
                    output, exactly as DL+DiReCT builds them (build_seg_maps
                    reproduces DiReCT.py's construction). The solve is then the
                    stock one and only the propagation is new. This is the
                    conservative default on purpose.
                    'surface-pv' builds BOTH boundaries as surfaces and
                    rasterises them as partial volume -- MEASURED WORSE on
                    containment at every smoothing level, see
                    surface_seg. The GM surface comes from the topology-
                    corrected ribbon, so sulci whose CSF fell below detection
                    are open. It supplies its own white surfaces, so surf_dir /
                    surfaces / build_white are ignored -- `topology` is not, it
                    selects the correction used to build them.
    wm_from_surface 'logits' only. False (default) solves on the segmentation
                    DiReCT would have used. True replaces the WM label with the
                    white surface's interior first, so the field's inner
                    boundary coincides with the surface that rides it -- see
                    pial_clean.prepare.
    propagate_on    'cuda' keeps the field in GPU memory from solve to surface;
                    'cpu' uses the numpy reference implementation
    velocity        reuse a field instead of solving (tensor or [D,H,W,3] array)
    pin             pin the medial wall (needs softmax_seg.nii.gz + label_def.csv)
    compute_thickness
                    False skips the DiReCT thickness map (and with it the
                    THICKNESS_PRIOR velocity cap -- see solve_velocity_field_t).
                    'thickness' comes back None.
    report          print evaluate_surface metrics per hemisphere. Defaults to
                    `verbose`, but it costs ~4s per 130k vertices and scales with
                    vertex count, so pass report=False in a loop.

    Returns {'surfaces': {hemi: (pial_verts, faces)}, 'white': {hemi: (v, f)},
             'velocity', 'thickness', 'seg', 'ref_img', 'tovox', 'totkr'}.
    """
    import pandas as pd

    if str(segmentation).lower() == 'surface-pv':
        reused = None
        if reuse_white:
            reused = _load_white_for_reuse(reuse_white, prep_dir, hemis=('lh', 'rh'),
                                           nsmooth=nsmooth, topology=topology,
                                           crop=crop, verbose=verbose)
        if verbose:
            print('building the segmentation from surfaces%s...'
                  % (' (reusing ?h.white)' if reused else ''))
        sd = surface_seg.build_surface_segmentation(prep_dir, hemis=tuple(hemis),
                                                    nsmooth=nsmooth, crop=crop,
                                                    topology=topology,
                                                    wm_surfaces=reused,
                                                    correct_ribbon=correct_ribbon,
                                                    verbose=verbose)
        d = dict(seg=sd['seg'], gmT=sd['gmT'], wmT=sd['wmT'], ref_img=sd['ref_img'],
                 tovox=sd['tovox'], totkr=sd['totkr'], surfaces=sd['surfaces'],
                 prep_dir=prep_dir)
        return _solve_and_propagate(d, prep_dir, propagate_on, velocity, pin, out_dir,
                                    verbose, report, compute_thickness, dtype, device,
                                    velocity_sigma=velocity_sigma,
                                    solve_margin=solve_margin, write_white=write_white,
                                    stats=stats, subject_id=subject_id,
                                    white_build=_white_build_record(nsmooth, topology, crop),
                                    extra=dict(gm_surfaces=sd['gm_surfaces'],
                                               found_csf=sd['found_csf']))
    elif str(segmentation).lower() != 'logits':
        raise ValueError("segmentation must be 'logits' or 'surface-pv', got %r"
                         % (segmentation,))

    if build_white is None:
        build_white = surfaces is None and not surf_dir
    if build_white:
        if surfaces is not None:
            raise ValueError('build_white=True and surfaces= are mutually exclusive')
        if verbose:
            print('building the white surfaces (%d Taubin steps, %s topology)...'
                  % (nsmooth, topology))
        # Both hemispheres regardless of `hemis`: the WM label is reconciled
        # against their union. ref_img puts the vertices in the solve grid's
        # tkrRAS, which is what prepare() expects -- the label grid's differs by
        # half a voxel per odd axis (0.5mm on a 256^3 conform vs a cropped
        # grid), under prepare's frame-check threshold -- and crops the label
        # volume to the ribbon, which is 4.9x fewer voxels to correct and mesh.
        _g, _w, _ref = pc.load_gm_wm_probability(prep_dir)
        surfaces = wm_surface.build_white_surfaces(prep_dir, regions=('lh', 'rh'),
                                                   nsmooth=nsmooth, verbose=verbose,
                                                   topology=topology, ref_img=_ref,
                                                   crop=crop)
        surf_dir = None

    d = pc.prepare(prep_dir, surf_dir, hemis=tuple(hemis), surfaces=surfaces,
                   wm_from_surface=wm_from_surface)
    return _solve_and_propagate(d, prep_dir, propagate_on, velocity, pin, out_dir,
                                verbose, report, compute_thickness, dtype, device,
                                velocity_sigma=velocity_sigma,
                                solve_margin=solve_margin, write_white=write_white,
                                stats=stats, subject_id=subject_id)


def _solve_and_propagate(d, prep_dir, propagate_on, velocity, pin, out_dir,
                         verbose, report, compute_thickness, dtype, device,
                         velocity_sigma=pc.VELOCITY_SIGMA,
                         solve_margin=solve_grid.MARGIN,
                         write_white=True, stats=False, subject_id=None,
                         white_build=None, extra=None):
    """Shared tail: solve the field, propagate each hemisphere, report."""
    import pandas as pd
    outer = d
    report = verbose if report is None else report
    on_gpu = str(propagate_on).lower() == 'cuda'
    # Solve on the cerebrum plus a margin rather than on the brain-mask box:
    # smaller where the cerebellum was, LARGER where the tissue was against a
    # face with no background to stop the propagation against. Skipped when a
    # field is supplied, since that field defines its own grid.
    sub = None
    if solve_margin is not None and velocity is None:
        d, sub = solve_grid.tighten(d, margin=solve_margin, verbose=verbose)
    hemis = tuple(d['surfaces'])
    seg, tovox, totkr, ref_img = d['seg'], d['tovox'], d['totkr'], d['ref_img']

    thickness = None
    if velocity is None:
        if verbose:
            print('solving the velocity field (%d iterations)...' % pc.MAX_ITERATIONS)
        vel_t, thick_t, _dev = pc.solve_velocity_field_t(
            seg, d['gmT'], d['wmT'], ref_img, verbose=verbose, device=device,
            compute_thickness=compute_thickness, velocity_sigma=velocity_sigma)
        thickness = thick_t.squeeze().cpu().numpy() if thick_t is not None else None
        # Only leave the GPU if something actually needs the host copy.
        velocity = vel_t if (on_gpu and not out_dir) else pc.velocity_to_numpy(vel_t)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            # on the caller's grid, so the file matches the prep's other volumes
            img = nib.Nifti1Image(velocity if sub is None else sub.restore(velocity),
                                  outer['ref_img'].affine)
            img.header['xyzt_units'] = 10
            nib.save(img, os.path.join(out_dir, 'pial_Velocity.nii.gz'))
            if on_gpu:
                velocity = vel_t            # keep using the resident tensor

    soft_seg = id_map = None
    if pin:
        soft = os.path.join(prep_dir, 'softmax_seg.nii.gz')
        ldef = os.path.join(prep_dir, 'label_def.csv')
        if os.path.exists(soft) and os.path.exists(ldef):
            soft_seg = np.asarray(nib.load(soft).dataobj)
            if sub is not None:
                soft_seg = sub.apply(soft_seg)
            id_map = {r.LABEL: int(r.ID) for _, r in pd.read_csv(ldef).iterrows()}
        else:
            print('WARNING: no softmax_seg.nii.gz / label_def.csv; the medial wall will '
                  'not be pinned and will be dragged outward.', file=sys.stderr)

    out = {}
    for hemi in hemis:
        white, faces = d['surfaces'][hemi]
        pin_mask = None
        if soft_seg is not None:
            no_push = build_no_push_mask(white, faces, seg, soft_seg, id_map, tovox, rings=0)
            pin_mask = build_pin_mask(no_push, white, faces, seg, soft_seg, id_map, tovox,
                                      scope='medial-wall', rings=0)
            if verbose:
                print('%s: %d/%d vertices with no cortex to move into, %d pinned'
                      % (hemi, no_push.sum(), len(no_push), pin_mask.sum()))
        pial = propagate(white, faces, velocity, seg, tovox, totkr, ref_img=ref_img,
                         pin_mask=pin_mask, on=propagate_on, device=device, dtype=dtype)
        if sub is not None:
            pial = sub.to_parent(pial)
        out[hemi] = (pial, faces)
        if out_dir:
            vinfo = volume_info_from_image(outer['ref_img'], prep_dir)
            nib.freesurfer.io.write_geometry(os.path.join(out_dir, '%s.pial' % hemi),
                                             pial, faces, create_stamp=None,
                                             volume_info=vinfo)
            if write_white:
                # The surface the propagation STARTED from, in the same frame as
                # the pial beside it (outer, i.e. before solve_grid.tighten).
                # Under `surface-pv` it is built in memory and was previously
                # never written, so anything wanting white-vs-pial afterwards --
                # regional_stats, a distance comparison, freeview -- had to
                # rebuild it: binary mask, topology correction, signed distance,
                # levelset_to_mesh and 50 Taubin steps, ~35 s a subject, for a
                # mesh the solve already had. Writing it here costs one file.
                #
                # Note the file is float32 (FreeSurfer geometry always is) while
                # the in-memory mesh is float64. The quantisation is ~1e-5 mm at
                # these coordinates; a caller that needs the exact float64 mesh
                # should still rebuild rather than read.
                nib.freesurfer.io.write_geometry(
                    os.path.join(out_dir, '%s.white' % hemi),
                    np.asarray(outer['surfaces'][hemi][0], np.float64), faces,
                    create_stamp=None, volume_info=vinfo)
                if white_build:
                    # so a later --reuse-white can REFUSE a parameter mismatch
                    # rather than silently start from the wrong surface
                    import json as _json
                    with open(os.path.join(out_dir, WHITE_BUILD), 'w') as fh:
                        _json.dump(white_build, fh)
        if report:
            # against the caller's grid, so the numbers stay comparable
            m = evaluate_surface(outer['surfaces'][hemi][0], pial, faces,
                                 outer['seg'], outer['tovox'],
                                 no_push=pin_mask if (pin_mask is not None
                                                      and pin_mask.any()) else None)
            print('%s: displacement %.3f mm, crossed_csf %d, self-intersections %d, '
                  'flipped %.4f%%  [%s]'
                  % (hemi, m['mean_displacement_mm'], m['crossed_csf_count'],
                     m['self_intersections'], m['flipped_face_pct'], propagate_on))
    if sub is not None:
        # Everything leaves on the grid it arrived on. The field is brought back
        # to the host even when the propagation kept it resident: a sub-grid
        # tensor handed to a later call would be used against a caller-grid
        # segmentation, and nothing downstream would notice.
        if not isinstance(velocity, np.ndarray):
            velocity = pc.velocity_to_numpy(velocity)
        velocity = sub.restore(velocity)
        thickness = None if thickness is None else sub.restore(thickness)
    if stats:
        if not out_dir:
            raise ValueError('stats=True needs out_dir to write the CSVs into')
        from . import regional_stats
        # Everything handed over is on the CALLER's grid: `outer` predates
        # solve_grid.tighten, the pials were brought back by sub.to_parent, and
        # the field by sub.restore just above. Passing the tightened `d` or a
        # sub-grid field instead would mislabel every vertex, silently.
        vel_np = (velocity if isinstance(velocity, np.ndarray)
                  else pc.velocity_to_numpy(velocity))
        sid = subject_id or os.path.basename(os.path.normpath(prep_dir))
        if verbose:
            print('aggregating regional statistics...')
        regional_stats.compute(prep_dir, out_dir=out_dir, subject_id=sid,
                               hemis=tuple(out), sd=outer,
                               pials={h: out[h][0] for h in out},
                               velocity=vel_np, verbose=verbose)
    res = dict(surfaces=out, white=outer['surfaces'], velocity=velocity,
               thickness=thickness, seg=outer['seg'], ref_img=outer['ref_img'],
               tovox=outer['tovox'], totkr=outer['totkr'])
    if extra:
        res.update(extra)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--prep-dir', required=True)
    p.add_argument('--surf-dir', help='directory holding ?h.white; omit to build them')
    p.add_argument('--build-white', action='store_true',
                   help='build the white surfaces from the segmentation in-process')
    p.add_argument('--velocity-sigma', type=float, default=pc.VELOCITY_SIGMA,
                   help='ANTs -b, the velocity smoothing sigma (default %.2f)'
                        % pc.VELOCITY_SIGMA)
    p.add_argument('--segmentation', default='logits',
                   choices=['logits', 'surface-pv'],
                   help='surface-pv builds both boundaries as surfaces and rasterises '
                        'them; the GM surface comes from the topology-corrected ribbon')
    p.add_argument('--wm-from-surface', action='store_true',
                   help="'logits' only: replace the WM label with the white "
                        "surface's interior before solving. Off by default, so "
                        "the solve is the stock DL+DiReCT one.")
    p.add_argument('--topology', default='nighres', choices=['nighres', 'gpu', 'none'],
                   help='topology correction when building white surfaces '
                        '(default nighres)')
    p.add_argument('--nsmooth', type=int, default=wm_surface.NSMOOTH_DEFAULT,
                   help='Taubin steps for the white surface (default %d)'
                        % wm_surface.NSMOOTH_DEFAULT)
    p.add_argument('--solve-margin', type=int, default=solve_grid.MARGIN,
                   help='voxels of background guaranteed around the cerebrum for the '
                        'solve and the propagation (default %d). The supplied grid is '
                        'the brain mask bounding box with none, so the tissue touches '
                        'the faces; -1 keeps it as given' % solve_grid.MARGIN)
    p.add_argument('--out-dir')
    p.add_argument('--no-white', action='store_true',
                   help='do not write ?h.white beside ?h.pial')
    p.add_argument('--hemi', nargs='+', default=['lh', 'rh'], choices=['lh', 'rh'])
    p.add_argument('--propagate-on', default='cuda', choices=['cpu', 'cuda'])
    p.add_argument('--thickness', action='store_true',
                   help='also compute the DiReCT thickness map. Off by default: it costs '
                        'two warp_image calls per integration point and two Gaussian '
                        'smooths per iteration (22.6s -> 13.3s without it), and on '
                        'validated data the velocity field is bit-identical either way '
                        'because the THICKNESS_PRIOR cap never binds')
    p.add_argument('--no-ribbon-correction', dest='correct_ribbon',
                   action='store_false',
                   help='surface-pv only: mesh the raw GM ribbon instead of running '
                        'the sub-threshold CSF detection on it')
    p.add_argument('--reuse-white',
                   help='directory (absolute, or a name inside --prep-dir) holding '
                        '?h.white from an earlier run to start from instead of '
                        'rebuilding the white surfaces')
    p.add_argument('--stats', action='store_true',
                   help='also write regional_stats\' result-thick-<metric>.csv into '
                        '--out-dir (field, field_raw, travel, nn, sym_nn). Reuses the '
                        'solve\'s segmentation, surfaces and field, so it costs the '
                        'aggregation alone rather than a 35s rebuild')
    p.add_argument('--subject', help='SUBJECT cell of the --stats CSVs '
                                     '(default: the prep directory name)')
    p.add_argument('--float64', action='store_true',
                   help='run the cuda propagation in double precision')
    p.add_argument('--check', action='store_true',
                   help='propagate BOTH ways off one solve and report the difference')
    args = p.parse_args()
    dtype = torch.float64 if args.float64 else torch.float32

    if not args.check:
        reconstruct(args.prep_dir, args.surf_dir, hemis=tuple(args.hemi),
                    propagate_on=args.propagate_on, out_dir=args.out_dir, dtype=dtype,
                    compute_thickness=args.thickness,
                    build_white=args.build_white or None, nsmooth=args.nsmooth,
                    topology=args.topology,
                    segmentation=args.segmentation,
                    velocity_sigma=args.velocity_sigma,
                    solve_margin=None if args.solve_margin < 0 else args.solve_margin,
                    write_white=not args.no_white,
                    stats=args.stats, subject_id=args.subject,
                    correct_ribbon=args.correct_ribbon)
        return

    import time
    # pin=False so both paths see identical inputs; the pin blend is exercised
    # separately below.
    r = reconstruct(args.prep_dir, args.surf_dir, hemis=tuple(args.hemi),
                    propagate_on='cpu', out_dir=None, verbose=True, pin=False,
                    build_white=args.build_white or None, nsmooth=args.nsmooth,
                    topology=args.topology,
                    segmentation=args.segmentation,
                    velocity_sigma=args.velocity_sigma,
                    solve_margin=None if args.solve_margin < 0 else args.solve_margin)
    for hemi in args.hemi:
        white, faces = r['white'][hemi]
        cpu = r['surfaces'][hemi][0]
        for dt, nm in ((torch.float32, 'float32'), (torch.float64, 'float64')):
            t = time.time()
            gpu = propagate(white, faces, r['velocity'], r['seg'], r['tovox'], r['totkr'],
                            ref_img=r['ref_img'], pin_mask=None, on='cuda', dtype=dt)
            dt_s = time.time() - t
            e = np.linalg.norm(gpu - cpu, axis=1)
            print('%s cuda/%s vs cpu: median %.3e  mean %.3e  max %.3e mm   (%.2fs)'
                  % (hemi, nm, np.median(e), e.mean(), e.max(), dt_s))
        # and once WITH the medial-wall pin, to exercise that branch too
        import pandas as pd
        soft = os.path.join(args.prep_dir, 'softmax_seg.nii.gz')
        ldef = os.path.join(args.prep_dir, 'label_def.csv')
        if os.path.exists(soft) and os.path.exists(ldef):
            ss = np.asarray(nib.load(soft).dataobj)
            im = {row.LABEL: int(row.ID) for _, row in pd.read_csv(ldef).iterrows()}
            npsh = build_no_push_mask(white, faces, r['seg'], ss, im, r['tovox'], rings=0)
            pm = build_pin_mask(npsh, white, faces, r['seg'], ss, im, r['tovox'],
                                scope='medial-wall', rings=0)
            a = pc.propagate_pial(white, faces,
                                  pc.velocity_to_numpy(r['velocity'])
                                  if torch.is_tensor(r['velocity']) else r['velocity'],
                                  r['seg'], r['tovox'], r['totkr'], pin_mask=pm)
            b = propagate(white, faces, r['velocity'], r['seg'], r['tovox'], r['totkr'],
                          ref_img=r['ref_img'], pin_mask=pm, on='cuda', dtype=torch.float64)
            e = np.linalg.norm(b - a, axis=1)
            print('%s pinned, cuda/float64 vs cpu: median %.3e  max %.3e mm'
                  % (hemi, np.median(e), e.max()))


if __name__ == '__main__':
    main()
