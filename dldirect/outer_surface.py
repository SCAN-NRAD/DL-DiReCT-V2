#!/usr/bin/env python
"""A smooth outer surface of the cortex -- the envelope, with the sulci bridged.

This is the surface you want when the sulcal interior is not the object of
interest: the wrapping that follows the gyral crowns and passes over each sulcus
rather than descending into it. FreeSurfer writes the same idea as
?h.pial-outer-smoothed, which it uses as the reference for the local
gyrification index; this builds one from the DL+DiReCT segmentation.

Method
------
Morphological closing of the cortical ribbon by a ball of radius R:

    closing_R(M) = erode_R(dilate_R(M))

A sulcus narrower than 2R is bridged, because the dilation joins its two banks
and the erosion cannot reopen a gap that is no longer there; a gyral crown, with
free space around it, is restored to where it was. R therefore sets the width of
what counts as "a sulcus to bridge" and nothing else.

Both operations are done with exact Euclidean distance transforms rather than a
discrete structuring element, so R is a true radius in millimetres and is not
quantised to the voxel lattice:

    dilate_R(M) = { x : dist(x, M)  <= R }
    erode_R(D)  = { x : dist(x, ~D) >  R }

Cavities left inside (a bridged sulcus becomes an enclosed void) are filled, the
isosurface is taken at the half-level, and the mesh is Taubin-smoothed with the
same filter and step count the white surface uses.

TOPOLOGY IS NOT CORRECTED. The closing usually produces a simple envelope, but
nothing here guarantees genus 0 and the returned mesh may carry handles. Run
topology_gpu.correct_topology on the closed mask first if that matters.
"""

import numpy as np
import nibabel as nib
import pymeshlab
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from skimage import measure

from . import wm_labels

RADIUS_MM = 3.0            # ball radius; sulci narrower than 2R are bridged
NSMOOTH_DEFAULT = 60       # Taubin steps on the envelope


def close_by_ball(mask, radius_mm, voxel_size=(1.0, 1.0, 1.0)):
    """Morphological closing by a ball of `radius_mm`, via distance transforms.

    Exact in millimetres: no discrete structuring element, so a non-integer
    radius and anisotropic voxels are both handled correctly.
    """
    m = np.asarray(mask) > 0
    dil = distance_transform_edt(~m, sampling=voxel_size) <= radius_mm
    return distance_transform_edt(dil, sampling=voxel_size) > radius_mm


def cortical_envelope(seg, radius_mm=RADIUS_MM, voxel_size=(1.0, 1.0, 1.0),
                      labels=(2, 3), fill=True):
    """Boolean envelope of the cortical ribbon: closed, then cavities filled.

    `labels` are the seg values to include -- (2, 3) is GM plus WM, i.e. the
    whole cerebrum, which is what you want: closing GM alone would bridge the
    GM/WM interface as readily as a sulcus.
    """
    m = np.isin(np.asarray(seg), list(labels))
    env = close_by_ball(m, radius_mm, voxel_size)
    if fill:
        env = binary_fill_holes(env)
    return env


def mesh_envelope(env, affine, nsmooth=NSMOOTH_DEFAULT, step_size=1):
    """Isosurface of a boolean envelope -> (vertices in `affine`'s frame, faces).

    `affine` maps voxel indices to the frame the surface should live in -- pass
    the tkrRAS matrix (wm_surface.get_vox2ras_tkr) to match ?h.white.
    """
    pad = 2
    v, f, _, _ = measure.marching_cubes(np.pad(env.astype(np.float32), pad), level=0.5)
    v = v - pad
    v = nib.affines.apply_affine(affine, v)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=v, face_matrix=f), 'envelope')
    ms.meshing_invert_face_orientation()
    ms.apply_coord_taubin_smoothing(stepsmoothnum=nsmooth)
    m = ms.current_mesh()
    return np.asarray(m.vertex_matrix()), np.asarray(m.face_matrix())


def build_outer_surface(seg, affine, radius_mm=RADIUS_MM, voxel_size=(1.0, 1.0, 1.0),
                        nsmooth=NSMOOTH_DEFAULT, labels=(2, 3), verbose=True):
    """seg -> (vertices, faces) for the smooth outer cortical surface."""
    env = cortical_envelope(seg, radius_mm, voxel_size, labels=labels)
    v, f = mesh_envelope(env, affine, nsmooth=nsmooth)
    if verbose:
        m = np.isin(np.asarray(seg), list(labels))
        print('outer surface: R=%.1fmm, ribbon %d -> envelope %d voxels (+%.1f%%), '
              '%d vertices' % (radius_mm, int(m.sum()), int(env.sum()),
                               100 * (env.sum() / max(m.sum(), 1) - 1), len(v)))
    return v, f


def build_hemisphere_outer(seg_labelled, df_labels, affine, region, excluded,
                           radius_mm=RADIUS_MM, nsmooth=NSMOOTH_DEFAULT,
                           cortex_label=None, verbose=True):
    """One hemisphere's outer surface, from a parcellated label volume.

    Uses the same hemisphere label set as the WM fill, unioned with that
    hemisphere's cortex label, so the envelope covers the ribbon rather than
    the WM alone.
    """
    # The WM fill's labels are the 'Left-*' / 'Right-*' structures. A parcellated
    # aseg names the cortex per gyrus instead, as 'lh-*' / 'rh-*', and those
    # carry the ribbon -- without them the envelope wraps white matter, not
    # cortex. Take both families for this side, minus the same exclusions.
    ids = list(wm_labels.ribbon_labels(df_labels, region, excluded))
    if cortex_label is not None:
        ids.append(cortex_label)
    ids = sorted(set(ids))
    m = np.isin(np.asarray(seg_labelled), ids)
    env = close_by_ball(m, radius_mm)
    env = binary_fill_holes(env)
    v, f = mesh_envelope(env, affine, nsmooth=nsmooth)
    if verbose:
        print('%s outer surface: R=%.1fmm, %d -> %d voxels, %d vertices'
              % (region, radius_mm, int(m.sum()), int(env.sum()), len(v)))
    return v, f
