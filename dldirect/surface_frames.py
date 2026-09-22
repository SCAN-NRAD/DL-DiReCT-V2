"""Mapping this pipeline's surfaces into scanner RAS, and checking that it worked.

WHY THIS EXISTS

The surfaces this pipeline writes are in the tkrRAS frame of `mri/aparc.atlas
+aseg`, whose affine is built from dim/pixdim alone and therefore centres on the
array -- it carries no link to scanner space. Worse, `preparedata.py` conforms
the CROPPED segmentation up to 256^3, and nibabel's conform() re-centres on the
input's own field of view, so the conformed grid is NOT the grid a FreeSurfer
run of the same scan would use. On the development subject the two differ by
(-1, -6, -12) voxels.

Assuming "both are 256^3 tkrRAS, so they match" is therefore wrong, and it is
wrong in a way that is hard to notice:

  - Centroid or bounding-box comparisons are dominated by a few outlier
    vertices (our white includes filled subcortical structures) and gave a
    12mm "shift" on surfaces that were fine, and no shift on surfaces that
    were not.
  - Nearest-neighbour distance between two cortical surfaces DOES NOT detect
    a translation of ~1 sulcal spacing. Misaligned by 12mm, our pial still
    scored a 1.8mm median distance to FreeSurfer's, because a folded surface
    always has *some* vertex nearby -- on the wrong gyrus. That produced a
    complete set of plausible, meaningless numbers.

The conclusive test is tissue labels: sample an independent segmentation at the
surface's mapped coordinates. A white surface sits on the WM/cortex boundary
with ~0% background; a pial sits at the cortex/CSF boundary with ~0% WM. Those
cannot be faked by a shift. `check_surface_frame` below does exactly that.

`preparedata.py` records the true scanner affine of the conformed grid in
`mri/conform_vox2ras.txt` precisely so this is recoverable.

SINGLE-GRID PIPELINE. With `preparedata.py --space cropped` there is no
conform: `mri/aparc.atlas+aseg` is written on the cropped segmentation grid
itself, `conform_vox2ras.txt` is that grid's own affine, and the surfaces are
in that grid's tkrRAS. Everything in this module works unchanged for both
layouts, because each function reads the grid it needs from the run's own
`mri/`. The one thing that does NOT carry over between layouts is tkr
coordinates themselves: the conformed and cropped tkr frames differ by 0 or
0.5 mm per axis depending on the parity of each crop dimension (measured
(+0.5,-0.5,+0.5) on a 131x141x167 crop, (-0.5,0,0) on 129x142x172), so
compare old and new outputs only through `our_surface_to_world`. See
doc/cropped-space-pipeline.md.
"""

import os
import numpy as np
import nibabel as nib


def _tkr(shape, zooms):
    """FreeSurfer tkrRAS for a volume, from dim/pixdim alone (as the surfaces use)."""
    ds = np.asarray(zooms[:3], float)
    ns = np.asarray(shape[:3], float) * ds / 2.0
    return np.array([[-ds[0], 0, 0, ns[0]],
                     [0, 0, ds[2], -ns[2]],
                     [0, -ds[1], 0, ns[1]],
                     [0, 0, 0, 1]], float)


def our_surface_to_world(prep_dir, ref='mri/aparc.atlas+aseg.nii.gz'):
    """4x4 mapping THIS pipeline's surface coordinates to scanner RAS.

    Needs `mri/conform_vox2ras.txt`, written by preparedata.py. Raises if it is
    absent rather than silently falling back to the identity, which is the
    mistake this module exists to prevent.
    """
    p = os.path.join(prep_dir, 'mri', 'conform_vox2ras.txt')
    if not os.path.exists(p):
        raise FileNotFoundError(
            "%s not found: without it the surfaces cannot be placed in scanner "
            "RAS. Re-run preparedata.py, or map nothing -- do not assume the "
            "tkr frames coincide." % p)
    im = nib.load(os.path.join(prep_dir, ref))
    return np.loadtxt(p) @ np.linalg.inv(_tkr(im.shape, im.header.get_zooms()))


def freesurfer_surface_to_world(fs_subject_dir, ref='mri/orig.mgz'):
    """4x4 mapping a FreeSurfer subject's surface coordinates to scanner RAS."""
    im = nib.load(os.path.join(fs_subject_dir, ref))
    return im.affine @ np.linalg.inv(im.header.get_vox2ras_tkr())


def cortex_labels(label_img):
    """Cortical label values actually present in `label_img`.

    Hardcoding is what broke this check: its default was the aseg convention
    (3, 42), while its companion our_surface_to_world defaults to
    mri/aparc.atlas+aseg, which uses the aparc convention (1001-2035) and
    contains no 3 or 42 at all. The check therefore reported cortex=0.0% and
    FAIL for every surface, including correct ones -- a guard that always cries
    wolf, which is worse than none.
    """
    img = nib.load(label_img) if isinstance(label_img, str) else label_img
    u = np.unique(np.asarray(img.dataobj))
    if 3 in u or 42 in u:
        return tuple(int(x) for x in (3, 42) if x in u)
    hi = tuple(int(x) for x in u if x >= 1000)
    if hi:
        return hi
    raise ValueError("no cortical labels found in the reference volume: it has "
                     "neither aseg 3/42 nor any aparc label >= 1000. Pass "
                     "`cortex=` explicitly.")


def check_surface_frame(verts, to_world, label_img, kind='white',
                        wm=(2, 41), cortex=None, min_tissue=None):
    """Sample `label_img` at mapped `verts` and say whether the frame is right.

    Catches the failure this module exists for: a whole-surface translation, of
    the kind that put our surfaces 12mm from FreeSurfer's earlier in this work.
    It is NOT a sub-voxel accuracy check -- a 1-2mm error passes, by design.

    The statistic is the fraction of vertices landing in tissue (WM or cortex).
    Measured on rh of sub-POBHC0002 with the surface deliberately shifted:

        shift      white wm+ctx      pial wm+ctx
         0mm          84.1%            72.3%
         2mm          82.6%            69.7%
         5mm          75.5%            66.0%
         8mm          67.7%            61.8%
        12mm          58.1%            53.8%
        20mm          41.4%            38.4%

    so the defaults (white 0.72, pial 0.62) pass a correctly placed surface with
    ~12 points of margin and fail at 8mm and beyond. `background` alone does not
    work as the criterion: a correct white surface already samples 12.5%
    background, well above the 5% the previous version demanded.

    `cortex=None` derives the cortical labels from the reference volume rather
    than assuming a convention -- see cortex_labels().

    Returns (ok, stats) with stats in percent.
    """
    img = nib.load(label_img) if isinstance(label_img, str) else label_img
    A = np.asarray(img.dataobj)
    if cortex is None:
        cortex = cortex_labels(img)
    if min_tissue is None:
        min_tissue = 0.72 if kind == 'white' else 0.62
    vox = nib.affines.apply_affine(np.linalg.inv(img.affine) @ to_world, verts)
    i = np.clip(np.rint(vox).astype(int), 0, np.array(A.shape) - 1)
    lab = A[i[:, 0], i[:, 1], i[:, 2]]
    in_wm = np.isin(lab, wm).mean()
    in_ctx = np.isin(lab, cortex).mean()
    st = dict(wm=100 * in_wm, cortex=100 * in_ctx,
              background=100 * (lab == 0).mean(),
              tissue=100 * (in_wm + in_ctx),
              in_bounds=100 * np.all((vox >= 0) & (vox < np.array(A.shape)), axis=1).mean())
    ok = (st['tissue'] >= 100 * min_tissue) and st['in_bounds'] > 99.0
    return ok, st


def volume_info_from_prep(prep_dir, ref='mri/aparc.atlas+aseg.nii.gz'):
    """FreeSurfer surface `volume_info` describing the grid our surfaces live in.

    A surface file should state its own frame. FreeSurfer's do: they carry the
    volume geometry (dimensions, voxel size, direction cosines and `cras`, the
    scanner-RAS coordinate of the volume centre) so any reader can place the
    surface without outside knowledge. Ours were written with
    `volume_info=None`, which is why freeview reports "Did not find any volume
    info" for them and why placing them correctly needed conform_vox2ras.txt.

    Pass the result to `nibabel.freesurfer.io.write_geometry(..., volume_info=)`.
    """
    p = os.path.join(prep_dir, 'mri', 'conform_vox2ras.txt')
    if not os.path.exists(p):
        return None                      # nothing truthful to say; say nothing
    M = np.loadtxt(p)
    im = nib.load(os.path.join(prep_dir, ref))
    shape = np.asarray(im.shape[:3], int)
    zooms = np.asarray(im.header.get_zooms()[:3], float)
    # cras: scanner-RAS coordinate of the volume centre (FreeSurfer's convention)
    cras = nib.affines.apply_affine(M, shape / 2.0)
    # direction cosines: the affine's columns, unit length
    cos = M[:3, :3] / zooms[None, :]
    return dict(head=np.array([2, 0, 20]), valid='1  # volume info valid',
                filename=os.path.join(prep_dir, ref),
                volume=shape, voxelsize=zooms,
                xras=cos[:, 0], yras=cos[:, 1], zras=cos[:, 2], cras=cras)
