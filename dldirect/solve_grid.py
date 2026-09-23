"""Run the solve on the cerebrum plus a margin, not on the whole cropped grid.

Two facts about the grid DL+DiReCT hands us, both measured over 18 subjects:

  * It is the bounding box of the hd-bet BRAIN mask with NO margin at all
    (crop.get_crop takes an exact bounding box). The cerebral tissue the solve
    actually acts on therefore reaches to within 0-2 voxels of five of the six
    faces, and touches the face outright on several.

  * On the sixth face it is 24-29 slices too big, because the brain mask
    includes the cerebellum and brainstem while the solve does not: its tissue
    is Left/Right-Cerebral-Cortex and -Cerebral-White-Matter only.

The first is a correctness problem. The pial is propagated by sampling the
velocity field at the vertex position with `padding_mode='border'`, so a vertex
that reaches the edge keeps reading the last in-grid voxel's velocity for the
rest of the integration -- there is no background there to stop it against, and
nothing reports that it happened. The field itself is also wrong within a
kernel radius of the face, since the Gaussians in the solve are normalised
against a domain that has been truncated.

The second is waste: ~18% of the voxels in every iteration of a 45-iteration
solve hold no cerebrum.

This module fixes both at once, by re-gridding onto

    bbox(seg > 0) grown by `margin` voxels

which is smaller than the supplied grid where the cerebellum was and LARGER
where the tissue was against a face -- the box is allowed to extend past the
original extent and the shortfall is filled with background, which is what the
region outside the brain mask always was. Measured over the 18 subjects, the
net is 3.11M -> 2.69M voxels (84%) while going from 0-2 voxels of margin to
exactly `margin` on all six faces.

Everything is an axis-aligned integer shift, so no interpolation happens in
either direction. Surfaces are mapped through the two grids' tkrRAS frames with
an exact affine (the frames differ by the shift and by half a voxel per axis
whose extent changed parity), and volumes are restored to the caller's grid
before they are returned, so the sub-grid is invisible from outside.
"""

import numpy as np
import nibabel as nib

from .compare_surfaces import tkr_to_world
from .field_pial_prototype import get_vox2ras_tkr, make_transforms

MARGIN = 2       # voxels of background guaranteed around the tissue on every face


class SubGrid(object):
    """An axis-aligned crop-and-pad of `ref_img`'s grid, with exact transforms.

    `lo` is the sub-grid's origin in the parent's voxel indices and MAY be
    negative; `shape` may exceed the parent's. Regions with no parent voxel are
    filled with `fill` (0, i.e. background).
    """

    def __init__(self, ref_img, lo, shape):
        self.parent = ref_img
        self.parent_shape = tuple(int(s) for s in ref_img.shape[:3])
        self.lo = np.asarray(lo, int)
        self.shape = tuple(int(s) for s in shape)
        affine = ref_img.affine.copy()
        affine[:3, 3] = nib.affines.apply_affine(ref_img.affine, self.lo.astype(float))
        hdr = ref_img.header.copy()
        hdr.set_data_shape(self.shape)
        self.img = nib.Nifti1Image(np.zeros(self.shape, np.float32), affine, hdr)
        self.tovox, self.totkr = make_transforms(self.img)
        # tkrRAS of the parent grid -> tkrRAS of this one, via scanner RAS.
        self.M = (np.linalg.inv(tkr_to_world(self.img, self.img.affine))
                  @ tkr_to_world(ref_img, ref_img.affine))
        self.Minv = np.linalg.inv(self.M)
        # the overlapping box, in each grid's own indices
        a = np.maximum(self.lo, 0)
        b = np.minimum(self.lo + np.array(self.shape), np.array(self.parent_shape))
        self._src = tuple(slice(int(x), int(y)) for x, y in zip(a, b))
        self._dst = tuple(slice(int(x), int(y)) for x, y in zip(a - self.lo, b - self.lo))

    # ---------------------------------------------------------------- volumes
    def apply(self, arr, fill=0):
        """A parent-grid array on this grid; missing voxels become `fill`."""
        arr = np.asarray(arr)
        out = np.full(self.shape + arr.shape[3:], fill, arr.dtype)
        out[self._dst] = arr[self._src]
        return out

    def restore(self, arr, fill=0):
        """A this-grid array back on the parent's grid."""
        arr = np.asarray(arr)
        out = np.full(self.parent_shape + arr.shape[3:], fill, arr.dtype)
        out[self._src] = arr[self._dst]
        return out

    # --------------------------------------------------------------- surfaces
    def to_sub(self, verts):
        v = np.asarray(verts)
        return (self.M[:3, :3] @ v.T).T + self.M[:3, 3]

    def to_parent(self, verts):
        v = np.asarray(verts)
        return (self.Minv[:3, :3] @ v.T).T + self.Minv[:3, 3]


def tight_box(active, shape, margin=MARGIN):
    """(lo, shape) of `active`'s bounding box grown by `margin`, unclamped."""
    nz = np.array(np.nonzero(active))
    if not nz.size:
        raise ValueError('nothing active; cannot choose a solve grid')
    lo = nz.min(1) - margin
    hi = nz.max(1) + 1 + margin
    return lo, tuple(int(x) for x in (hi - lo))


def tighten(d, margin=MARGIN, verbose=True):
    """Re-grid a prepare()/build_surface_segmentation() dict onto the tight box.

    Returns (new_dict, SubGrid), or (d, None) when the grid is already exactly
    right so there is nothing to do.
    """
    ref_img = d['ref_img']
    lo, shape = tight_box(np.asarray(d['seg']) > 0, ref_img.shape[:3], margin)
    if tuple(lo) == (0, 0, 0) and shape == tuple(ref_img.shape[:3]):
        return d, None
    sub = SubGrid(ref_img, lo, shape)
    out = dict(d)
    for k in ('seg', 'gmT', 'wmT', 'pv_gm', 'pv_wm'):
        if d.get(k) is not None:
            out[k] = sub.apply(d[k])
    out['ref_img'], out['tovox'], out['totkr'] = sub.img, sub.tovox, sub.totkr
    for k in ('surfaces', 'gm_surfaces'):
        if d.get(k) is not None:
            out[k] = {h: (sub.to_sub(v), f) for h, (v, f) in d[k].items()}
    if verbose:
        n0 = int(np.prod(ref_img.shape[:3]))
        n1 = int(np.prod(shape))
        print('solve grid: %s -> %s (%.2fM -> %.2fM voxels, %.0f%%), %d-voxel margin'
              % (tuple(ref_img.shape[:3]), shape, n0 / 1e6, n1 / 1e6,
                 100.0 * n1 / n0, margin))
    return out, sub
