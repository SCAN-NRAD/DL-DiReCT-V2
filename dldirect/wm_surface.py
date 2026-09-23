"""Build the topologically-correct, smoothed WM surface from a segmentation.

This is the mesh half of dl_wm_surface_parallel_dev.py, factored out so the
surface can be built in-process -- pial_pipeline.reconstruct() calls it when no
white surface is supplied, which removes the round trip through ?h.white on
disk. The script remains the CLI and keeps the file writing; both go through
build_hemisphere() here, so there is one implementation of the geometry.

Original: Victor B. B. Mello, 09/2024, SCAN / University of Bern.
"""

import numpy as np
import nibabel as nib
import pymeshlab

from . import wm_labels

NSMOOTH_DEFAULT = 50          # Taubin steps; the shell pipelines pass -ns 50
CROP_MARGIN = 8               # voxels of background kept around the ribbon when cropping


def get_vox2ras_tkr(t1):
    """Transformation for FreeView visualisation.

    Derived from the header's pixdim/dim rather than the affine, so it is the
    tkrRAS of whatever grid the segmentation was written on.
    """
    ds = t1.header._structarr['pixdim'][1:4]
    ns = t1.header._structarr['dim'][1:4] * ds / 2.0
    v2rtkr = np.array([[-ds[0], 0, 0, ns[0]],
                       [0, 0, ds[2], -ns[2]],
                       [0, -ds[1], 0, ns[1]],
                       [0, 0, 0, 1]], dtype=np.float32)

    return v2rtkr


def rec_surf(binary, affine, r):
    # Reconstruct topologically correct surfaces
    # https://nighres.readthedocs.io/en/latest/shape/topology_correction.html
    # https://nighres.readthedocs.io/en/latest/surface/levelset_to_mesh.html
    # Ref: Bazin and Pham (2007). Topology correction of segmented medical images using a fast marching algorithm doi:10.1016/j.cmpb.2007.08.006
    # nighres reads only dims and zooms from this image and returns voxel
    # coordinates, so it does not need the 256^3 conform; this runs unchanged on
    # the cropped grid (preparedata.py --space cropped). 'background->object'
    # propagation does need background around the object, which the crop
    # (the brain mask's bounding box) always leaves: measured >= 4 voxels of
    # margin on every face for both hemispheres. Guard it anyway with an
    # integer pad that is subtracted from the vertices -- exact, no resampling.
    # Note the correction is NOT invariant to how much zero padding surrounds
    # the object (it is deterministic for a given array): on sub-POBHC0002 rh
    # the 256^3 conform and the bare crop disagree on 35 of 264k object voxels,
    # all boundary voxels along the medial side, which moves ~260 of 145k
    # vertices by more than 0.1mm (mean point-to-surface 0.0015mm). So pad only
    # when the object actually touches a face.
    import nighres                       # heavy; only needed to build a surface
    pad = 0
    touches = any(b[0].any() or b[-1].any() for b in
                  (binary, np.moveaxis(binary, 1, 0), np.moveaxis(binary, 2, 0)))
    if touches:
        pad = 2
        print('WARNING: %s WM mask touches the volume border; padding by %d voxels for topology correction' % (r, pad))
        binary = np.pad(binary, pad)
    farray_img = nib.Nifti1Image(binary.astype(np.float64), affine)

    propag = 'background->object'
    connect = '6/18'
    minimum_distance = 1e-5

    ret = nighres.shape.topology_correction(farray_img, 'binary_object', minimum_distance=minimum_distance, propagation=propag,connectivity=connect)
    l2m_ret = nighres.surface.levelset_to_mesh(ret['corrected'], connectivity=connect)
    vertices = l2m_ret['result']['points'] - pad
    faces = l2m_ret['result']['faces']

    return vertices, faces


def signed_distance(mask, workers=2):
    """Exact signed Euclidean distance to the boundary of `mask`.

    The two transforms are independent and scipy's distance_transform_edt
    releases the GIL, so they run concurrently.

    MEASURED AND REJECTED: a narrow chamfer band computed with GPU min-pools is
    61x faster (5.9s -> 0.1s) but is NOT a substitute. Chamfer makes a diagonal
    neighbour distance 1 where Euclidean makes it sqrt(2), which moves the
    zero crossing by up to half a voxel and changes the mesh: on lh the
    isosurface came back with the same vertex count but a different
    triangulation and chi -42 instead of 2. Widening the band does not help --
    the error is sub-voxel, not far-field. An exact GPU transform
    (Felzenszwalb separable) would work; a chamfer one will not.
    """
    from concurrent.futures import ThreadPoolExecutor
    from scipy.ndimage import distance_transform_edt
    m = np.asarray(mask) > 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = ex.submit(distance_transform_edt, ~m)
        inn = ex.submit(distance_transform_edt, m)
        return (out.result() - inn.result()).astype(np.float32)


def rec_surf_gpu(binary, affine, r, pad=2, priority=None, n_bands=8):
    """rec_surf's contract, with the topology correction done on the GPU.

    topology_gpu.correct_topology grows a genus-0 seed by adding only simple
    points, which is ~6.6x faster than nighres' sequential fast-marching front
    (71.8s -> 11.0s per hemisphere measured over 36 PoB hemispheres, all 36
    genus 0 and single-bodied). The corrected mask is turned into a signed
    distance levelset and handed to the SAME nighres mesher, because
    levelset_to_mesh is connectivity-consistent and skimage's marching cubes is
    not -- measured, it reports spurious handles on a set whose digital Euler
    characteristic is exactly 1.

    The '26-6' pair is the one that yields chi=2 through that mesher; the dual
    gives chi=-34 with 10 bodies, despite nighres being called with '6/18'.

    Padding is unconditional here and even, so the subfield parity classes the
    growth uses are unchanged by it; the vertices come back in the unpadded
    frame. Against nighres on the same fills the resulting white surfaces agree
    to a median of 6e-06 mm and a p95 of 0.012 mm, differing only at the sites
    where the two choose to cut a handle.
    """
    import nighres
    from .topology_gpu import correct_topology

    b = np.pad(np.asarray(binary) > 0, pad)
    pr = None if priority is None else np.pad(np.asarray(priority, np.float32), pad,
                                              constant_values=1.0)
    corrected, info = correct_topology(b, verbose=False, pair='26-6',
                                       priority=pr, n_bands=n_bands)
    levelset = signed_distance(corrected)
    l2m = nighres.surface.levelset_to_mesh(nib.Nifti1Image(levelset, affine),
                                           connectivity='6/18')
    return l2m['result']['points'] - pad, l2m['result']['faces']


def rec_surf_raw(binary, affine, r, pad=2):
    """rec_surf's contract with NO topology correction: mesh the mask as it is.

    For a baseline that wants the original DiReCT's inputs rather than this
    pipeline's. The result is not genus 0 and may have several bodies -- that is
    the point of the arm, not a defect -- so nothing downstream may assume a
    sphere. It still goes through nighres' levelset_to_mesh rather than
    skimage's marching cubes, for the reason rec_surf_gpu gives: that mesher is
    connectivity-consistent and skimage's is not.
    """
    import nighres
    b = np.pad(np.asarray(binary) > 0, pad)
    levelset = signed_distance(b)
    l2m = nighres.surface.levelset_to_mesh(nib.Nifti1Image(levelset, affine),
                                           connectivity='6/18')
    return l2m['result']['points'] - pad, l2m['result']['faces']


def hemisphere_binary(seg, df_labels, region, excluded):
    """The filled WM mask for one hemisphere.

    Regions based on dl output label_def.csv. The excluded set comes from the
    record preparedata.py wrote for THIS run (wm_labels), not from a second
    copy of the list: the two used to disagree -- filled.mgz dropped the
    hippocampus but this mask kept it, so the surface produced here enclosed
    a structure the pipeline believed it had excluded. Every label that is
    not excluded is deliberately filled: the subcortical structures are what
    make the WM mask simply connected for topology_correction.
    """
    if region == 'lh':
        labels = wm_labels.hemisphere_labels(df_labels, 'Left', excluded)
    elif region == 'rh':
        labels = wm_labels.hemisphere_labels(df_labels, 'Right', excluded)
    else:
        raise ValueError('not a valid region: %r' % (region,))
    mask = np.isin(seg, labels)
    return np.array(np.where(mask, 1, 0), dtype=np.int32)


def build_hemisphere(seg, df_labels, affine, region, excluded, nsmooth=NSMOOTH_DEFAULT,
                     topology='nighres', priority=None):
    """Segmentation -> (vertices, faces) for one hemisphere's white surface.

    Topology correction, marching cubes, the tkrRAS affine, then `nsmooth`
    Taubin steps at pymeshlab's defaults. Vertices come back in the tkrRAS of
    the grid `affine` describes.

    topology='nighres'  the shipped sequential correction
    topology='none'     no correction at all: the mask is meshed as it stands.
                        For the original-DiReCT baseline. NOT genus 0, possibly
                        several bodies.
    topology='gpu'      rec_surf_gpu. Validated at 36 hemispheres for TOPOLOGY
                        (36/36 genus 0, single body) and for geometry away from
                        the handle cuts (p95 0.012mm). NOT the default: the cut
                        sites differ from nighres' by up to ~2.8mm per
                        hemisphere and the resulting pial surfaces have not been
                        compared, which is the test this project judges a
                        default change by.
    """
    binary = hemisphere_binary(seg, df_labels, region, excluded)
    if topology == 'gpu':
        vertices, faces = rec_surf_gpu(binary, affine, region, priority=priority)
    elif topology == 'nighres':
        vertices, faces = rec_surf(binary, affine, region)
    elif topology in ('none', None, False):
        vertices, faces = rec_surf_raw(binary, affine, region)
    else:
        raise ValueError('topology must be "nighres", "gpu" or "none", got %r'
                         % (topology,))

    # apply affine for FS visualization and matching with the MRI
    transf_vertx = nib.affines.apply_affine(affine, vertices)
    # get a smooth mesh
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=transf_vertx, face_matrix=faces), region)
    ms.meshing_invert_face_orientation()
    ms.apply_coord_taubin_smoothing(stepsmoothnum=nsmooth)

    m = ms.current_mesh()
    return m.vertex_matrix(), m.face_matrix()


def _build_one(job):
    seg, df_labels, affine, region, excluded, nsmooth, topology, priority = job
    return region, build_hemisphere(seg, df_labels, affine, region, excluded, nsmooth,
                                    topology=topology, priority=priority)


def ribbon_bbox(seg, df_labels, excluded, margin=CROP_MARGIN,
                regions=('lh', 'rh')):
    """Slices bounding both hemispheres' ribbon, grown by `margin` voxels.

    `margin` is background the operations downstream need around the object:
    the topology correction's pad (2), the signed distance transform, and the
    6 mm ball closing in surface_seg.locality_guard. 8 covers all three. The
    box is clamped to the array, so a mask that already reaches a face simply
    gets less margin there -- the callers that care pad for themselves.
    """
    from . import wm_labels
    m = np.isin(seg, sorted(set(sum((list(wm_labels.ribbon_labels(df_labels, r, excluded))
                                     for r in regions), []))))
    nz = np.array(np.nonzero(m))
    if not nz.size:
        raise ValueError('the ribbon is empty; nothing to crop to')
    lo = np.maximum(nz.min(1) - margin, 0)
    hi = np.minimum(nz.max(1) + 1 + margin, np.array(seg.shape[:3]))
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def load_inputs(prep_dir, crop=False, margin=CROP_MARGIN):
    """(seg, df_labels, affine, excluded, label_img) from a prep directory.

    Whatever grid preparedata.py wrote the segmentation on (256^3 conform by
    default, the cropped grid with --space cropped) is the grid the surfaces
    are built on, and the tkrRAS returned is that grid's. `label_img` is the
    image that grid belongs to -- pass it to retarget_surface.tkr_to_tkr as
    `src_ref` to map the resulting vertices anywhere else.

    crop=True first cuts the label volume down to the ribbon's bounding box
    plus `margin` (ribbon_bbox). The cut is a plain array slice with the
    affine's origin moved by the same integer offset: nothing is resampled and
    no anatomy is lost. On a 256^3 conform this is a 4.9x reduction in voxels
    (16.78M -> ~3.46M), and the white-surface build is dominated by three costs
    that are all proportional to volume -- measured on one hemisphere, topology
    correction 8.05 -> 2.51 s, distance transform 3.46 -> 0.42 s, meshing
    2.50 -> 0.77 s, i.e. 14.0 s -> 3.7 s.

    It is NOT bit-identical, because the topology correction is not invariant
    to how much background surrounds the object (already noted in rec_surf).
    Measured against the uncropped build on lh: median surface distance
    0.0001 mm, p95 0.0002 mm, max 4.47 mm -- the same shape everywhere except
    at the handful of sites where the two corrections choose to cut a handle
    differently, which is the same class and magnitude of difference as
    nighres-vs-GPU.
    """
    import os
    import pandas as pd
    seg_img = nib.load(os.path.join(prep_dir, 'mri', 'aparc.atlas+aseg.nii.gz'))
    seg = seg_img.get_fdata()
    df_labels = pd.read_csv(os.path.join(prep_dir, 'label_def.csv')) \
        .set_index('LABEL').to_dict()
    # The structures preparedata.py left out of the hemisphere fill for this
    # run. Read, not re-specified, so the two masks cannot disagree.
    excluded = wm_labels.read_record(os.path.join(prep_dir, 'mri'))
    if crop:
        sl = ribbon_bbox(seg, df_labels, excluded, margin)
        off = np.array([s.start for s in sl], float)
        seg = np.ascontiguousarray(seg[sl])
        aff = seg_img.affine.copy()
        aff[:3, 3] = nib.affines.apply_affine(seg_img.affine, off)
        seg_img = nib.Nifti1Image(seg.astype(np.float32), aff, seg_img.header)
        seg_img.header.set_data_shape(seg.shape)
    affine = get_vox2ras_tkr(seg_img)
    return seg, df_labels, affine, excluded, seg_img


def build_white_surfaces(prep_dir, regions=('lh', 'rh'), nsmooth=NSMOOTH_DEFAULT,
                         parallel=True, verbose=True, topology='nighres', priority=None,
                         ref_img=None, crop=None):
    """{region: (vertices, faces)} built from a prep directory.

    The two hemispheres are independent, so they run in a process pool by
    default -- with topology='nighres' the correction is the expensive part and
    it is single-threaded. With topology='gpu' the pool is NOT used, because the
    hemispheres would then contend for one device; they run in sequence.

    WHICH FRAME THE VERTICES COME BACK IN depends on `ref_img`:

      ref_img=None   the label grid's tkrRAS, i.e. the grid
                     mri/aparc.atlas+aseg.nii.gz is on. This is what the ?h.white
                     writer wants, since it stamps that grid's volume_info.
      ref_img=<img>  that image's tkrRAS. Pass the solve's reference (the one
                     load_gm_wm_probability returns) when the surfaces are going
                     into pial_clean.prepare, which documents its `surfaces=`
                     as being in the cropped frame. The two frames are NOT the
                     same: a 256^3 conform and a cropped grid with an odd extent
                     differ by half a voxel per odd axis, measured 0.5 mm on the
                     PoB preps, which is under prepare's 1.5 mm frame-check
                     threshold and so passes silently.

    `crop` cuts the label volume to the ribbon's bounding box first (see
    load_inputs); it defaults to True whenever `ref_img` is given, since a
    caller that is already mapping frames has no reason to pay for the conform.
    """
    crop = (ref_img is not None) if crop is None else crop
    seg, df_labels, affine, excluded, label_img = load_inputs(prep_dir, crop=crop)
    if verbose:
        print('WM fill excludes: %s' % ', '.join(excluded))
        if crop:
            print('label volume cropped to %s (%.2fM voxels)'
                  % (tuple(seg.shape), seg.size / 1e6))
    jobs = [(seg, df_labels, affine, r, excluded, nsmooth, topology, priority) for r in regions]
    if parallel and len(jobs) > 1 and topology != 'gpu':
        from multiprocessing.pool import Pool
        with Pool(len(jobs)) as pool:
            res = pool.map(_build_one, jobs)
    else:
        res = [_build_one(j) for j in jobs]
    out = {r: vf for r, vf in res}
    if ref_img is not None:
        from .retarget_surface import tkr_to_tkr
        M = tkr_to_tkr(prep_dir, ref_img, src_ref=label_img)
        out = {r: ((M[:3, :3] @ np.asarray(v).T).T + M[:3, 3], f)
               for r, (v, f) in out.items()}
    return out
