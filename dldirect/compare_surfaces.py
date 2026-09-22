"""Compare surfaces that live in different tkrRAS frames, via scanner RAS.

Why this is needed
------------------
A tkrRAS frame is built from an image's `dim`/`pixdim` alone, so it centres
on that image's own array. Two packages processing the same subject
therefore put their surfaces in *different* frames whenever their conformed
grids are centred differently, and vertex coordinates are not comparable
between them even though both describe the same head.

Concretely, for the FreeSurfer subject `bert`:

    our conformed grid, world affine t   = (134.4, -122, 134)
    bert recon-all orig.mgz, affine t    = (133.4, -110, 128)
    difference                           = (  1.0,  -12,   6)

which is exactly the offset that earlier work here had fitted empirically
and hardcoded as `T = (1, -12, 6)`. It is not a fitted constant and not a
property of FreeSurfer; it is the difference between two field-of-view
centres, and this module derives it rather than measuring it.

The route between any two surfaces is scanner RAS (world coordinates):

    world  =  X.affine @ inv(vox2ras_tkr(X)) @ v_tkr

where `X` is the volume the surface was built on. Composing that mapping
one way and its inverse the other takes vertices from one package's frame
into the other's with no fitted term anywhere.

Two prerequisites, both of which are easy to get wrong:

1. `crop.py` must shift the affine by the crop offset. Without that fix
   every cropped volume claims its first voxel sits at the template origin,
   and `world` above is wrong by `affine[:3,:3] @ crop_origin` -- on bert,
   by (-61, 24, -52) mm.
2. `nibabel.processing.conform()` re-centres on the input's own field of
   view, which *discards* the world position of the grid it produces. Our
   `mri/aparc.atlas+aseg.nii.gz` is deliberately written with the tkr
   affine, so it cannot be used to recover world coordinates. The conformed
   grid's true world affine is saved separately by `preparedata.py` as
   `mri/conform_vox2ras.txt`; this module reads that.

Note that conform()'s re-centring is also the reason the crop fix does not
move our surfaces: a pure translation of the input affine is absorbed by
the re-centring, and the conformed voxel array is byte-identical either way
(verified on bert). The fix changes only the recorded world affine, which
is precisely what this comparison needs.

Reporting
---------
Distances are reported symmetrically. A nearest-*vertex* distance between
two independently-tessellated meshes has a floor set by their vertex
spacing -- measured at ~0.315 mm for these meshes -- so a residual at that
scale means agreement, not disagreement. Point-to-*surface* distance is
also reported, which does not have that floor.
"""
import argparse
import os

import numpy as np
import nibabel as nib
from scipy import spatial


def get_vox2ras_tkr(img):
    """FreeSurfer-style tkrRAS transform built from an image's own header.
    Same construction as dl_wm_surface_parallel_dev.py, so that surfaces
    written by this pipeline are interpreted in the frame they were made in."""
    # Taken from zooms/shape rather than the NIfTI-only pixdim/dim fields, so
    # that MGH volumes (FreeSurfer's orig.mgz) work too. For a 1 mm LIA 256^3
    # volume this reproduces FreeSurfer's own tkr matrix exactly.
    ds = np.asarray(img.header.get_zooms()[:3], dtype=np.float64)
    ns = np.asarray(img.shape[:3], dtype=np.float64) * ds / 2.0
    return np.array([[-ds[0], 0, 0, ns[0]],
                     [0, 0, ds[2], -ns[2]],
                     [0, -ds[1], 0, ns[1]],
                     [0, 0, 0, 1]], dtype=np.float64)


def tkr_to_world(img, world_affine=None):
    """Matrix taking tkrRAS coordinates of `img`'s grid into scanner RAS.

    `world_affine` overrides `img.affine`, which is required for our own
    conformed volumes: they are written carrying the tkr affine, so their
    world position has to come from `mri/conform_vox2ras.txt` instead.
    """
    A = img.affine if world_affine is None else world_affine
    return A @ np.linalg.inv(get_vox2ras_tkr(img))


def load_our_surface(subject_dir, surf_name):
    """Load a surface produced by this pipeline, with its tkr->world matrix.

    Reads the world affine recorded by preparedata.py. If it is absent the
    run predates that change; rerun preparedata.py rather than guessing, as
    any substituted affine would silently reintroduce a fitted offset.
    """
    ref_path = os.path.join(subject_dir, 'mri', 'aparc.atlas+aseg.nii.gz')
    aff_path = os.path.join(subject_dir, 'mri', 'conform_vox2ras.txt')
    if not os.path.exists(aff_path):
        raise SystemExit(
            "%s not found.\nThis run predates the world-affine record. Rerun\n"
            "    python -m dldirect.preparedata -inputpath %s\n"
            "(with the crop.py affine fix present) to regenerate it."
            % (aff_path, subject_dir))
    ref = nib.load(ref_path)
    verts, faces = nib.freesurfer.io.read_geometry(os.path.join(subject_dir, 'surf', surf_name))
    return verts, faces, tkr_to_world(ref, np.loadtxt(aff_path))


def load_freesurfer_surface(fs_subject_dir, surf_name, ref_name='orig.mgz'):
    """Load a recon-all surface with its tkr->world matrix. FreeSurfer writes
    orig.mgz with a genuine scanner affine, so no override is needed."""
    ref = nib.load(os.path.join(fs_subject_dir, 'mri', ref_name))
    verts, faces = nib.freesurfer.io.read_geometry(os.path.join(fs_subject_dir, 'surf', surf_name))
    return verts, faces, tkr_to_world(ref)


def _point_to_surface(points, verts, faces):
    """Nearest distance from each point to the triangle mesh, which unlike a
    nearest-vertex distance has no vertex-spacing floor.

    Deliberately does NOT fall back to nearest-vertex on failure. Doing so
    reports the vertex distance under the surface-distance label, which is
    the same number wearing a different name -- and it is the *smaller*,
    floor-limited quantity, so it flatters the comparison.
    """
    try:
        import trimesh
    except ImportError as exc:
        raise SystemExit("--point-to-surface needs trimesh: %s" % exc)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    try:
        return np.abs(trimesh.proximity.signed_distance(mesh, points))
    except Exception as exc:
        raise SystemExit(
            "--point-to-surface failed (%s: %s).\n"
            "trimesh's proximity queries need the 'rtree' package, which is not\n"
            "installed in this environment. Install it (pip install rtree) or drop\n"
            "--point-to-surface and read the nearest-vertex numbers against the\n"
            "~0.315 mm floor noted in the output." % (type(exc).__name__, exc))


def compare(a_verts, a_faces, a_to_world, b_verts, b_faces, b_to_world,
            point_to_surface=False):
    """Symmetric distances between two surfaces, computed in scanner RAS."""
    A = nib.affines.apply_affine(a_to_world, a_verts)
    B = nib.affines.apply_affine(b_to_world, b_verts)

    d_ab = spatial.cKDTree(B).query(A)[0]
    d_ba = spatial.cKDTree(A).query(B)[0]
    out = {
        'n_a': len(A), 'n_b': len(B),
        'centroid_offset_mm': B.mean(0) - A.mean(0),
        'a_to_b_mean': float(d_ab.mean()), 'a_to_b_median': float(np.median(d_ab)),
        'a_to_b_p95': float(np.percentile(d_ab, 95)),
        'b_to_a_mean': float(d_ba.mean()), 'b_to_a_median': float(np.median(d_ba)),
        'b_to_a_p95': float(np.percentile(d_ba, 95)),
        'symmetric_mean': float(0.5 * (d_ab.mean() + d_ba.mean())),
        'hausdorff95': float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95))),
    }
    if point_to_surface:
        p_ab = _point_to_surface(A, B, b_faces)
        p_ba = _point_to_surface(B, A, a_faces)
        out['a_to_b_surface_mean'] = float(p_ab.mean())
        out['b_to_a_surface_mean'] = float(p_ba.mean())
        out['symmetric_surface_mean'] = float(0.5 * (p_ab.mean() + p_ba.mean()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--subject-dir', required=True,
                   help='our pipeline output directory (contains mri/, surf/)')
    p.add_argument('--fs-subject-dir', required=True,
                   help='FreeSurfer recon-all subject directory to compare against')
    p.add_argument('--hemi', nargs='+', default=['lh', 'rh'], choices=['lh', 'rh'])
    p.add_argument('--surf', default='white',
                   help="surface to compare on our side (default: white)")
    p.add_argument('--fs-surf', default=None,
                   help="surface on the FreeSurfer side (default: same as --surf)")
    p.add_argument('--point-to-surface', action='store_true',
                   help='also report point-to-triangle distance (no vertex-spacing floor)')
    args = p.parse_args()

    fs_surf = args.fs_surf or args.surf

    for hemi in args.hemi:
        av, af, aw = load_our_surface(args.subject_dir, '%s.%s' % (hemi, args.surf))
        bv, bf, bw = load_freesurfer_surface(args.fs_subject_dir, '%s.%s' % (hemi, fs_surf))
        m = compare(av, af, aw, bv, bf, bw, point_to_surface=args.point_to_surface)

        print('=== %s: ours %s (%d verts) vs FreeSurfer %s (%d verts) ==='
              % (hemi, args.surf, m['n_a'], fs_surf, m['n_b']))
        print('  derived tkr->tkr translation : %s mm'
              % np.round((bw[:3, 3] - aw[:3, 3]), 3))
        print('  centroid offset in world RAS : %s mm' % np.round(m['centroid_offset_mm'], 3))
        print('  ours->FS   mean %.3f  median %.3f  p95 %.3f mm'
              % (m['a_to_b_mean'], m['a_to_b_median'], m['a_to_b_p95']))
        print('  FS->ours   mean %.3f  median %.3f  p95 %.3f mm'
              % (m['b_to_a_mean'], m['b_to_a_median'], m['b_to_a_p95']))
        print('  symmetric mean %.3f mm | 95%% Hausdorff %.3f mm'
              % (m['symmetric_mean'], m['hausdorff95']))
        if args.point_to_surface:
            print('  point-to-surface symmetric mean %.3f mm' % m['symmetric_surface_mean'])
        print('  (nearest-vertex floor for independently tessellated meshes ~0.315 mm)')


if __name__ == '__main__':
    main()
