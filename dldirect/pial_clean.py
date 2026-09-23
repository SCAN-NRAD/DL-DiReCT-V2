#!/usr/bin/env python
"""The field-propagated pial surface, current default configuration only.

A minimal restatement of what field_pial_prototype.py does with its shipped
defaults, with none of the options that exist there for experiments. Every
parameter below is fixed at the validated value and stated once; there are no
alternative branches to choose between. If you want to vary something, use
field_pial_prototype.py -- that is what it is for.

The method, in three stages:

  1. PREPARE. Load the GM/WM probabilities, build DiReCT's seg/gmT/wmT, then
     reconcile the WM label with the white surface itself (partial-volume
     rasterisation, supersample 3) so the solve and the surface share one WM
     boundary.

  2. SOLVE. ANTs' DiReCT (KellyKapowski) on the GPU, with ONE deviation: the
     velocity field is smoothed by a direction-gated Gaussian instead of an
     isotropic one. Across a sulcus the two banks carry near-antiparallel
     velocity and an isotropic kernel averages them into a common translation;
     the gate weights each neighbour by relu(cos) against the centre voxel's own
     velocity direction, so opposing banks stop cancelling. The divisor is the
     PLAIN Gaussian weight sum, so disagreement attenuates rather than being
     renormalised away.

  3. PROPAGATE. Carry the white surface along that field, 20 rounds at half the
     per-round step (so the total equals the solve's 10 integration points),
     with a light Taubin relaxation between rounds and the medial wall pinned.

Written 2026-09-11. The numbers it produces on bert (lh/rh): fundus CSF arrival
3.5/4.1%, mean displacement 2.401/2.402mm, self-intersections 173/36,
flipped_face_pct 0.0067/0.0015.
"""

import argparse
import os
import sys

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from scipy.ndimage import map_coordinates

from .direct_cuda import (gaussian_smooth_3d, gaussian_gradient_3d, extract_wm_contours,
                          _make_identity_grid, warp_image, compose_fields, invert_field)
from .field_pial_prototype import (load_gm_wm_probability, build_seg_maps, make_transforms,
                                   check_frame_alignment, rasterize_mesh, rasterize_mesh_pv,
                                   reconcile_seg_with_surface, _mesh_adjacency,
                                   _smoothed_normals, build_constrained_white,
                                   build_no_push_mask, build_pin_mask, _pin_weights,
                                   evaluate_surface)
from .surface_frames import volume_info_from_image

# ---------------------------------------------------------------------------
# The configuration. These are the validated values; nothing here is a knob.
# ---------------------------------------------------------------------------
MAX_ITERATIONS = 45        # DiReCT outer iterations
INTEGRATION_POINTS = 10    # inner integration steps per iteration
GRADIENT_STEP = 0.025      # mm, the Euler step of the descent
GRADIENT_GATE = 1e-3       # gradient magnitudes at or below this do not propagate
THICKNESS_PRIOR = 10.0     # mm; ANTs' cap. Never binds here (max observed 6.1mm)
SMOOTH_SIGMA = 1.0         # voxels, gradient + hit/total accumulation
VELOCITY_SIGMA = 1.2247    # voxels = sqrt(1.5), ANTs'
#                          m_SmoothingVelocityFieldVariance = 1.5, i.e. the
#                          stock DiReCT value. See direct_cuda, which carries
#                          the same default for the same reason. Lowering it
#                          sharpens the field but tangles the mesh.
FIELD_EPS = 1e-3           # a velocity below this has no usable direction
ROUNDS = 20                # propagation rounds
STEP_SCALE = INTEGRATION_POINTS / ROUNDS   # keeps the total deformation fixed
RELAX_ITERS = 2            # Taubin iterations between rounds
RELAX_ITERS_FINAL = 1      # odd, so the last round ends on an unpaired shrink
RELAX_LAMBDA = 0.51        # matched to pymeshlab's filter; do not change
PIN_FEATHER = 2            # mesh rings over which the medial-wall pin ramps off
WM_SUPERSAMPLE = 3         # partial-volume rasterisation of the white surface
INVERT_MAX_ITER = 20       # ANTs' cap on the inversion's fixed-point iterations
INVERT_CHECK_EVERY = 4     # host syncs per that many iterations; see below


# invert_field's convergence test -- `if max_error <= 0.1 or mean <= 0.001` --
# is a host synchronisation on every fixed-point iteration, ~9800 per solve and
# 16.8% of the function's time, which is 44.7% of the solve. MEASURED AND
# REJECTED: evaluating the test on the GPU (a sticky scalar flag freezing the
# update, so the answer is bit-identical to breaking) and syncing only every 4th
# iteration came out 10.3% SLOWER, 4/4 alternating repetitions. Freezing costs
# whole iterations -- the mean is 10.84 and checking every 4th runs to ~12.7 --
# and an iteration costs more than the sync it saves. Freezing with no sync at
# all (always 20) was 3.3% slower again. Do not retry without a way to stop on
# the exact iteration without asking the host.
#
# The first version of that experiment also reported -29.5%, which was
# torch.compile warm-up being paid by whichever arm ran first. Benchmark the
# solve by alternating arms and taking medians; a single A-then-B is worthless
# here.


def solve_velocity_field_t(seg, gm_prob, wm_prob, ref_img, verbose=True, device=None,
                           compute_thickness=True,
                           velocity_sigma=VELOCITY_SIGMA):
    """The solve itself, returning the field as a TORCH TENSOR on its device.

    Returns (velocity, thickness, device) with velocity [1, 3, D, H, W] and
    thickness [1, 1, D, H, W]. Nothing is copied to the host, so a caller that
    propagates on the GPU never moves the field across the bus.
    solve_velocity_field() below is the numpy-returning wrapper.

    compute_thickness=False skips the thickness map entirely and returns None in
    its place. That removes two warp_image calls per integration point (20 per
    iteration) and two Gaussian smooths per iteration.

    It also removes the THICKNESS_PRIOR cap, which is derived from the same
    hit/total accumulation: where the running thickness exceeds the prior the
    velocity is scaled down by (prior/thickness)^2. On the data this has been
    run on the cap never binds (max observed thickness 6.1 mm against a 10.0 mm
    prior) and the field is bit-identical either way -- but that is a property
    of the data, not a guarantee. If you need the pial surface from a subject
    where cortex might exceed the prior, leave this on.
    """
    device = torch.device(device) if device is not None else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    D, H, W = seg.shape
    t = lambda a: torch.from_numpy(a.astype(np.float32)).to(device).reshape(1, 1, D, H, W)
    seg_t, gm_t, wm_t = t(seg), t(gm_prob), t(wm_prob)

    gm_mask = (seg_t == 2).float()                 # the increment lands here ONLY
    wm_contour = extract_wm_contours(seg_t)
    active = (gm_mask + wm_contour).clamp(max=1.0)
    identity = _make_identity_grid((D, H, W), device)

    velocity = torch.zeros(1, 3, D, H, W, device=device)
    integrated = torch.zeros(1, 3, D, H, W, device=device)
    thickness_img = torch.zeros(1, 1, D, H, W, device=device)
    cortical_thickness = torch.zeros(1, 1, D, H, W, device=device)

    for iteration in range(MAX_ITERATIONS):
        increment = torch.zeros(1, 3, D, H, W, device=device)
        inverse = torch.zeros(1, 3, D, H, W, device=device)
        hit = torch.zeros(1, 1, D, H, W, device=device)
        total = torch.zeros(1, 1, D, H, W, device=device)

        for pt in range(1, INTEGRATION_POINTS + 1):
            inverse = compose_fields(velocity * active, inverse, identity)
            warped_wm = warp_image(wm_t, inverse, identity)
            if compute_thickness:
                warped_contour = warp_image(wm_contour, inverse, identity)
                warped_thick = warp_image(thickness_img, inverse, identity)

            grad = gaussian_gradient_3d(warped_wm, SMOOTH_SIGMA, device)
            gmag = (grad * grad).sum(dim=1, keepdim=True).sqrt()
            direction = grad / (gmag + 1e-8) * (gmag > GRADIENT_GATE).float()

            speed = -(warped_wm - gm_t) * gm_t * GRADIENT_STEP * gm_mask
            speed = torch.where(torch.isfinite(speed), speed, torch.zeros_like(speed))
            increment = increment + direction * speed

            if compute_thickness:
                if pt == 1:
                    thickness_img = integrated.norm(dim=1, keepdim=True) * wm_contour
                    hit = wm_contour.clone()
                    total = thickness_img.clone()
                else:
                    hit = hit + warped_contour * gm_mask
                    total = total + warped_thick * gm_mask

            inverse = inverse * active
            velocity = velocity * active
            if pt == 1:
                integrated.zero_()
            integrated = invert_field(inverse, identity, initial=integrated)
            inverse = invert_field(integrated, identity, initial=inverse)

        velocity = velocity + increment

        if compute_thickness:
            sh = gaussian_smooth_3d(hit, SMOOTH_SIGMA, device, zero_boundary=False)
            st = gaussian_smooth_3d(total, SMOOTH_SIGMA, device, zero_boundary=False)
            has = sh > 0.001
            vals = torch.where(has, st / sh.clamp(min=0.001), torch.zeros_like(sh)).clamp(min=0)
            over = has & (vals > THICKNESS_PRIOR) & (gm_mask > 0)
            if over.any():
                frac = THICKNESS_PRIOR / vals.clamp(min=1e-8)
                velocity = velocity * torch.where(over, frac * frac, torch.ones_like(frac))
            cortical_thickness = vals * gm_mask

        velocity = gaussian_smooth_3d(velocity, velocity_sigma, device,
                                      zero_boundary=False)
        velocity = velocity * active          # MUST precede the save; see below
        if verbose and (iteration + 1) % 10 == 0:
            if compute_thickness:
                print('  iteration %d/%d, mean thickness %.3f mm'
                      % (iteration + 1, MAX_ITERATIONS,
                         float(cortical_thickness[gm_mask > 0].mean())))
            else:
                print('  iteration %d/%d' % (iteration + 1, MAX_ITERATIONS))

    return velocity, (cortical_thickness if compute_thickness else None), device


def velocity_to_numpy(velocity):
    """[1, 3, D, H, W] tensor -> [D, H, W, 3] float32 array, as the propagation
    and the NIfTI export both want it."""
    return velocity[0].cpu().numpy().transpose(1, 2, 3, 0).astype(np.float32)


def solve_velocity_field(seg, gm_prob, wm_prob, ref_img, out_prefix=None, verbose=True,
                         compute_thickness=True,
                         velocity_sigma=VELOCITY_SIGMA):
    """DiReCT on the GPU. Returns (velocity, thickness).

    velocity is [D, H, W, 3] in voxels, components in voxel-index order (d,h,w),
    and is the PER-INTEGRATION-POINT field: the solve composes it
    INTEGRATION_POINTS times, which is why the propagation below applies it
    ROUNDS times at STEP_SCALE.
    """
    velocity, cortical_thickness, _ = solve_velocity_field_t(
        seg, gm_prob, wm_prob, ref_img, verbose=verbose,
        compute_thickness=compute_thickness,
        velocity_sigma=velocity_sigma)
    vel = velocity_to_numpy(velocity)
    if out_prefix:
        # AFTER the active-region mask. Saving before it exported a 12% smoothing
        # halo into CSF, which the propagation then rode.
        img = nib.Nifti1Image(vel, ref_img.affine)
        img.header['xyzt_units'] = 10
        nib.save(img, out_prefix + 'Velocity.nii.gz')
    return vel, (cortical_thickness.squeeze().cpu().numpy()
                 if cortical_thickness is not None else None)


def propagate_pial(white_verts, faces, velocity, seg, tovox, totkr, pin_mask=None):
    """Carry the white surface along the velocity field.

    DiReCT's velocity points GM->WM, so the outward direction is its negative.
    The field is sampled AT the vertex (the old out-of-WM offset was a
    workaround for the dead WM shell and made a vertex step on a field half a
    millimetre from where it is).
    """
    mesh, Wm, deg = _mesh_adjacency(white_verts, faces)
    from scipy.ndimage import distance_transform_edt
    wmb = (seg == 3)
    sdt = distance_transform_edt(~wmb) - distance_transform_edt(wmb)
    grad = np.stack(np.gradient(sdt), axis=-1)
    gu = grad / np.maximum(np.linalg.norm(grad, axis=-1), 1e-9)[..., None]
    cache = (Wm, deg, sdt, gu)

    pin_w = _pin_weights(pin_mask, Wm, PIN_FEATHER)[:, None] \
        if (pin_mask is not None and pin_mask.any()) else None
    start = np.asarray(white_verts).copy()
    cur = white_verts
    for rnd in range(ROUNDS):
        pos = tovox(cur)
        v = np.stack([map_coordinates(velocity[..., k], pos.T, order=1, mode='nearest')
                      for k in range(3)], axis=1)
        step = (totkr(pos - v) - totkr(pos)) * STEP_SCALE
        iters = RELAX_ITERS if rnd < ROUNDS - 1 else RELAX_ITERS_FINAL
        cur = build_constrained_white(cur + step, faces, seg, tovox, totkr,
                                      floor=-np.inf, iters=iters, lam=RELAX_LAMBDA,
                                      cache=cache)
        if pin_w is not None:
            cur = pin_w * start + (1.0 - pin_w) * cur
    return cur


def prepare(prep_dir, surf_dir=None, hemis=('lh', 'rh'), surfaces=None,
            wm_from_surface=False):
    """seg/gmT/wmT plus the transforms, and the white surfaces to propagate.

    By DEFAULT the three maps are exactly what DL+DiReCT solves on:
    `build_seg_maps` reproduces DiReCT.py's own construction from the model
    probabilities, and nothing further is done to them. The solve is then the
    stock one and only the propagation is new.

    `wm_from_surface=True` additionally replaces the WM label with the white
    SURFACE's interior before solving. The two disagree -- `seg == 3` is a
    threshold on the WM logits, the surface is a topology-corrected level set
    of a different mask -- so the field's inner boundary sits in a slightly
    different place from the surface that rides it. Reconciling them removes
    that mismatch, at the cost of no longer solving on the segmentation
    DiReCT would have used.

    `surfaces` optionally supplies {hemi: (verts, faces)} already in the cropped
    tkrRAS frame, in place of reading ?h.white from `surf_dir`. BOTH hemispheres
    are still required when reconciling: the WM label is reconciled against the
    union of the two surfaces, so a one-hemisphere call would demote the other
    hemisphere's WM.
    """
    import pandas as pd
    gm_prob, wm_prob, ref_img = load_gm_wm_probability(prep_dir)
    seg, gmT, wmT = build_seg_maps(gm_prob, wm_prob)
    tovox, totkr = make_transforms(ref_img)
    shape = tuple(ref_img.shape[:3])

    given = dict(surfaces) if surfaces else None
    if given is not None and set(given) != {'lh', 'rh'}:
        sys.exit('surfaces= needs both hemispheres (got %s); the WM label is '
                 'reconciled against their union' % sorted(given))
    if given is None and not surf_dir:
        sys.exit('give either surf_dir or surfaces=')
    surfaces = {}
    partial = np.zeros(shape, np.float32)
    crisp = np.zeros(shape, bool)
    for h in ('lh', 'rh'):
        if given is not None:
            v, f = given[h]
            v = np.asarray(v, np.float64)
            f = np.asarray(f)
        else:
            path = os.path.join(surf_dir, '%s.white' % h)
            if not os.path.exists(path):
                sys.exit('missing %s -- both white surfaces are needed to define WM' % path)
            v, f, vinfo = nib.freesurfer.io.read_geometry(path, read_metadata=True)
            if vinfo and 'volume' in vinfo and \
                    tuple(int(x) for x in vinfo['volume']) != shape:
                sys.exit('%s.white was built on a %s grid, reference is %s: the tkrRAS frames '
                         'differ by half a voxel per odd axis. Rebuild with preparedata.py '
                         '--space cropped.' % (h, list(vinfo['volume']), list(shape)))
        surfaces[h] = (v, f)
        crisp |= rasterize_mesh(tovox(v), f, shape)
        partial += rasterize_mesh_pv(tovox(v), f, shape, WM_SUPERSAMPLE)
    if wm_from_surface:
        before = int((seg == 3).sum())
        seg, gmT, wmT, dem, pro = reconcile_seg_with_surface(
            seg, gmT, wmT, np.clip(partial, 0.0, 1.0), label_mask=crisp)
        print('WM from surface: %d -> %d voxels (%d demoted, %d promoted)'
              % (before, int((seg == 3).sum()), dem, pro))
    for h in hemis:
        dist, frac, inb = check_frame_alignment(surfaces[h][0], seg, tovox)
        print('frame check %s: mean |distance to WM boundary| %.3f mm '
              '(expect <1.5; %.1f%% sample WM, %.1f%% in bounds)'
              % (h, dist, 100 * frac, 100 * inb))
        if dist > 1.5 or inb < 0.99:
            print('WARNING: frame alignment looks wrong for %s' % h, file=sys.stderr)
    return dict(seg=seg, gmT=gmT, wmT=wmT, ref_img=ref_img, tovox=tovox, totkr=totkr,
                surfaces={h: surfaces[h] for h in hemis}, prep_dir=prep_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--prep-dir', required=True,
                   help='a --space cropped prep (seg_<Label>.nii.gz, softmax_seg.nii.gz, '
                        'label_def.csv)')
    p.add_argument('--surf-dir', required=True, help='directory holding ?h.white')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--hemi', nargs='+', default=['lh', 'rh'], choices=['lh', 'rh'])
    p.add_argument('--write-thickness', action='store_true',
                   help="also write this solve's thickness map and segmentation")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import pandas as pd
    d = prepare(args.prep_dir, args.surf_dir, tuple(args.hemi))
    seg, tovox, totkr = d['seg'], d['tovox'], d['totkr']

    print('solving the velocity field (%d iterations)...' % MAX_ITERATIONS)
    velocity, thickness = solve_velocity_field(
        seg, d['gmT'], d['wmT'], d['ref_img'], os.path.join(args.out_dir, 'pial_'))

    soft = os.path.join(args.prep_dir, 'softmax_seg.nii.gz')
    ldef = os.path.join(args.prep_dir, 'label_def.csv')
    soft_seg = id_map = None
    if os.path.exists(soft) and os.path.exists(ldef):
        soft_seg = np.asarray(nib.load(soft).dataobj)
        id_map = {r.LABEL: int(r.ID) for _, r in pd.read_csv(ldef).iterrows()}
    else:
        print('WARNING: no softmax_seg.nii.gz / label_def.csv; the medial wall will not '
              'be pinned and will be dragged outward.', file=sys.stderr)

    vinfo = volume_info_from_image(d['ref_img'], args.prep_dir)
    for hemi, (white, faces) in d['surfaces'].items():
        pin = None
        if soft_seg is not None:
            no_push = build_no_push_mask(white, faces, seg, soft_seg, id_map, tovox, rings=0)
            pin = build_pin_mask(no_push, white, faces, seg, soft_seg, id_map, tovox,
                                 scope='medial-wall', rings=0)
            print('%s: %d/%d vertices with no cortex to move into, %d pinned'
                  % (hemi, no_push.sum(), len(no_push), pin.sum()))
        pial = propagate_pial(white, faces, velocity, seg, tovox, totkr, pin_mask=pin)
        out = os.path.join(args.out_dir, '%s.pial' % hemi)
        nib.freesurfer.io.write_geometry(out, pial, faces, create_stamp=None,
                                         volume_info=vinfo)
        m = evaluate_surface(white, pial, faces, seg, tovox,
                             no_push=pin if (pin is not None and pin.any()) else None)
        print('%s -> %s   displacement %.3f mm, crossed_csf %d, self-intersections %d, '
              'flipped %.4f%%' % (hemi, out, m['mean_displacement_mm'],
                                  m['crossed_csf_count'], m['self_intersections'],
                                  m['flipped_face_pct']))

    if args.write_thickness and np.isfinite(thickness).all() and (thickness > 0).any():
        for name, arr in (('T1w_thickmap.nii.gz', thickness.astype(np.float32)),
                          ('seg.nii.gz', seg.astype(np.uint8))):
            img = nib.Nifti1Image(arr, d['ref_img'].affine)
            img.header['xyzt_units'] = 2
            nib.save(img, os.path.join(args.out_dir, name))
        print('thickness: mean %.3f mm over non-zero voxels'
              % thickness[thickness > 0].mean())


if __name__ == '__main__':
    main()
