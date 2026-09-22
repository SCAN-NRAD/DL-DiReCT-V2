import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from sklearn.svm import LinearSVC
from nibabel.processing import conform

# For FS visualization
def get_vox2ras_tkr(t1):
    # Transformation for FreeView visualization
    ds = t1.header._structarr['pixdim'][1:4]
    ns = t1.header._structarr['dim'][1:4] * ds / 2.0
    v2rtkr = np.array([[-ds[0], 0, 0, ns[0]],
                       [0, 0, ds[2], -ns[2]],
                       [0, -ds[1], 0, ns[1]],
                       [0, 0, 0, 1]], dtype=np.float32)
                       
    return v2rtkr

# Parser for the shell script
parser = argparse.ArgumentParser()
parser.add_argument('-inputpath', '--input')
parser.add_argument('--space', choices=('conformed', 'cropped'), default='conformed',
                    help="grid the mri/ volumes (and hence the surfaces) are built on. "
                         "'conformed' (default): resample to LIA 256^3 as FreeSurfer's "
                         "mris_make_surfaces requires. 'cropped': stay on the cropped "
                         "grid the segmentation, DiReCT and the velocity field already "
                         "live on, so the whole FreeSurfer-free path has ONE grid and "
                         "one tkrRAS frame. See doc/cropped-space-pipeline.md.")
args = parser.parse_args()

# Read DL+DiReCT results
# segmentation
seg_img = nib.load(args.input+'/softmax_seg.nii.gz')
# MRI 
brain_img = nib.load(args.input+'/T1w_norm_noskull_cropped.nii.gz')
if args.space == 'conformed':
    # Conform (256x256x256). Measured on sub-POBHC0002 (crop 131x141x167):
    # conform() is a pure integer shift of (62, 57, 44) voxels with zero
    # resampling loss (99.88% of voxels identical to the shifted input, the
    # rest being the corpus-callosum relabel below), because rescale_affine
    # maps input voxel (S-1)//2 onto output voxel 127. The hazard is not the
    # resampling but the FRAME: the tkrRAS of each grid centres on S/2, so for
    # an odd crop dimension the two grids' tkr frames differ by exactly half a
    # voxel. A consumer that reads the surfaces (built here, in the 256^3 tkr
    # frame) as if they were in the cropped grid's tkr frame places every
    # vertex 0.5 voxel off per odd axis -- which is what field_pial_prototype
    # did (0.866 voxel in total on that subject, all three dims odd).
    seg_image = conform(seg_img, order=0, orientation = 'LIA')
    brain_image = conform(brain_img, order=0, orientation = 'LIA')
else:
    # No conform: the cropped grid IS the reference grid. Nothing below
    # resamples, so the surfaces built from mri/aparc.atlas+aseg share the
    # voxel grid of seg_<Label>.nii.gz, softmax_seg.nii.gz, the thickness map
    # and the DiReCT velocity field, and the single tkrRAS frame derived from
    # this grid's dim/pixdim is the one every consumer uses.
    seg_image = seg_img
    brain_image = brain_img
# Write the mri/ volumes with the grid's TRUE vox2ras, not the centred tkrRAS.
# The surfaces are built in tkrRAS either way -- that is derived from dim/pixdim
# and is unaffected by this -- but the volumes then state their real scanner
# position, so a surface carrying volume geometry (see surface_frames.py) and the
# volumes agree. Writing the tkr affine here made every mri/*.mgz claim
# cras=[0,0,0]; a surface that truthfully claims cras=[6.4,6,6] then sits 6mm off
# the volume, and FreeSurfer's mris_make_surfaces silently fails to deform the
# pial (measured: thickness 2.21mm -> 0.0025mm).
affine = seg_image.affine
tkr_affine = get_vox2ras_tkr(seg_image)

# DeepSCAN label definition
df_labels = pd.read_csv(args.input+'/label_def.csv').set_index('LABEL').to_dict()

# Transform DeepSCAN corpus callosum and WM-hypointensities label into L and R WM
output_data = seg_image.get_fdata()
label_ids = df_labels['ID']

# Not every model segments WM-hypointensities: it is part of the v6/v7 label sets,
# but not of the default v0 one. Merge whichever of the midline labels the model provides.
midline_labels = ['Corpus-Callosum']
if 'WM-hypointensities' in label_ids:
    midline_labels.append('WM-hypointensities')
else:
    print('WARNING: segmentation model used does not detect WM-hypointensities, '
          'surface reconstruction may be unreliable')

# Separate the hemispheres with a plane fitted to the labels that already carry a
# laterality, instead of splitting the corpus callosum bounding box in the middle.
# The bounding box approach used only the two extreme CC voxels, was quantised to
# whole voxels and was axis aligned, so it could not follow a tilted head.
LATERAL_PREFIXES = {'left': ('Left', 'lh'), 'right': ('Right', 'rh')}
MAX_FIT_VOXELS = 40000

def _side_labels(side):
    return [label_ids[lbl] for lbl in label_ids if lbl.startswith(LATERAL_PREFIXES[side])]

def _fit_hemisphere_plane(seg):
    # deterministic stride based subsampling, the fit does not need every voxel
    coords = {}
    for side in ('left', 'right'):
        pts = np.argwhere(np.isin(seg, _side_labels(side)))
        if pts.size == 0:
            raise SystemExit('ERROR: no {} sided labels were segmented, '
                             'cannot separate the hemispheres.'.format(side))
        coords[side] = pts[::max(1, len(pts) // MAX_FIT_VOXELS)]

    x = np.vstack([coords['left'], coords['right']]).astype(np.float64)
    y = np.concatenate([np.zeros(len(coords['left'])), np.ones(len(coords['right']))])

    return LinearSVC(C=0.01, max_iter=20000).fit(x, y)

midline_mask = np.isin(output_data, [label_ids[lbl] for lbl in midline_labels])
seg_img = output_data.copy()
midline_voxels = np.argwhere(midline_mask)
if midline_voxels.size > 0:
    # positive side of the decision function is the right hemisphere
    is_right = _fit_hemisphere_plane(output_data).decision_function(
        midline_voxels.astype(np.float64)) > 0
    seg_img[tuple(midline_voxels[is_right].T)] = label_ids['Right-Cerebral-White-Matter']
    seg_img[tuple(midline_voxels[~is_right].T)] = label_ids['Left-Cerebral-White-Matter']

# export the transformed segmentation
trans_seg_mgz = nib.freesurfer.mghformat.MGHImage(np.array(seg_img,dtype=np.int32) , affine, header=None, extra=None, file_map=None)
atlas_name = 'DKatlas' if np.max(seg_img) <= 3000 else '2009s'
nib.save(trans_seg_mgz,args.input+'/mri/aparc.'+atlas_name+'+aseg.mgz')
nib.save(trans_seg_mgz,args.input+'/mri/aparc.atlas+aseg.nii.gz')
nib.save(trans_seg_mgz,args.input+'/mri/aparc.atlas+aseg.mgz')

# The surfaces are built in the tkrRAS affine, which is built from dim/pixdim
# alone and so centres on the array. That discards the link to scanner RAS:
# conform() re-centres on the input's own field of view, so the true world
# position of the conformed grid survives only in seg_image.affine. Record it
# here so a consumer that needs world coordinates -- e.g. comparing our surfaces
# against another package's, which live in that package's own tkr frame -- can
# recover it without redoing the conform. Requires the crop.py affine fix to be
# correct. In --space cropped this is simply the cropped grid's own affine (the
# same one softmax_seg.nii.gz carries); the file keeps its name because
# surface_frames.py and compare_surfaces.py read it as "the scanner vox2ras of
# the grid mri/aparc.atlas+aseg.nii.gz is on", which is what it still is.
np.savetxt(args.input+'/mri/conform_vox2ras.txt', seg_image.affine)

# Everything from here to the normalised MRI exists only to feed FreeSurfer's
# mri_normalize / mri_edit_wm_with_aseg / mri_pretess / mris_make_surfaces
# (see fast_surface_reconstruction.sh), which require the 256^3 conform. No
# FreeSurfer-free consumer reads aseg.presurf.mgz, filled.mgz or wm.seg.mgz
# (grep: only that script does), so the cropped path does not write them.
if args.space == 'cropped':
    # crude normalization on the MRI, as below; kept because raw_brain.mgz is
    # what extract_morphometrics.py / smooth_pial.py sample and norm.mgz is the
    # volume the surfaces are checked against (WM/GM intensity midpoint).
    mri = np.array(brain_image.get_fdata(),dtype=np.int32)
    wm_region = np.where((seg_img==df_labels['ID']['Right-Cerebral-White-Matter']) | (seg_img==df_labels['ID']['Left-Cerebral-White-Matter']))
    knorm = 110/np.mean(mri[wm_region])
    mri_normalized = np.array( np.where(seg_img!=0, knorm*mri,0) ,dtype=np.int32)
    nib.save(nib.freesurfer.mghformat.MGHImage(mri, affine, header=None, extra=None, file_map=None), args.input+'/mri/raw_brain.mgz')
    brain_mgz = nib.freesurfer.mghformat.MGHImage(mri_normalized, affine, header=None, extra=None, file_map=None)
    nib.save(brain_mgz,args.input+'/mri/brain.mgz')
    nib.save(brain_mgz,args.input+'/mri/norm.mgz')
    raise SystemExit(0)

# change pial labels to create aseg.mgz
labels_lh = [df_labels['ID'][x] for x in df_labels['ID'].keys() if x.startswith('lh')]
mask_lh = np.isin(seg_img, labels_lh)
labels_rh = [df_labels['ID'][x] for x in df_labels['ID'].keys() if x.startswith('rh')]
mask_rh = np.isin(seg_img, labels_rh)
temp = np.array(np.where(mask_lh,3,seg_img),dtype=np.int32)
seg = np.array(np.where(mask_rh,42,temp),dtype=np.int32)
seg_mgz = nib.freesurfer.mghformat.MGHImage(seg, affine, header=None, extra=None, file_map=None)
nib.save(seg_mgz,args.input+'/mri/aseg.presurf.mgz')

# create filled
labels_wm_lh = [df_labels['ID'][x] for x in df_labels['ID'].keys() if (x.startswith('Left') and x != 'Left-Cerebellum' and x != 'Left-Hippocampus') ]
mask_wm_lh = np.isin(seg_img, labels_wm_lh)
labels_wm_rh = [df_labels['ID'][x] for x in df_labels['ID'].keys() if (x.startswith('Right') and x != 'Right-Cerebellum' and x != 'Right-Hippocampus') ]
mask_wm_rh = np.isin(seg_img, labels_wm_rh)
temp = np.array(np.where(mask_wm_lh, 255, 0),dtype=np.int32)
wm_fill = np.array(np.where(mask_wm_rh,127,temp),dtype=np.int32)
wm_fill_mgz = nib.freesurfer.mghformat.MGHImage(wm_fill, affine, header=None, extra=None, file_map=None)
nib.save(wm_fill_mgz,args.input+'/mri/filled.mgz')

# export WM seg
wm_seg = np.where( (seg == df_labels['ID']['Left-Cerebral-White-Matter']) | (seg == df_labels['ID']['Right-Cerebral-White-Matter']), 110, 0)
wmseg_mgz = nib.freesurfer.mghformat.MGHImage(np.array(wm_seg,dtype=np.uint8), affine, header=None, extra=None, file_map=None)
nib.save(wmseg_mgz,args.input+'/mri/wm.seg.mgz')

# crude normalization on the MRI
mri = np.array(brain_image.get_fdata(),dtype=np.int32)
wm_region = np.where((seg==df_labels['ID']['Right-Cerebral-White-Matter']) | (seg==df_labels['ID']['Left-Cerebral-White-Matter']))
knorm = 110/np.mean(mri[wm_region])
mri_normalized = np.array( np.where(seg!=0, knorm*mri,0) ,dtype=np.int32)
# raw MRI data
brain_mgz_raw = nib.freesurfer.mghformat.MGHImage(mri, affine, header=None, extra=None, file_map=None)
brain_mgz = nib.freesurfer.mghformat.MGHImage(mri_normalized, affine, header=None, extra=None, file_map=None)
nib.save(brain_mgz,args.input+'/mri/brain.mgz')
nib.save(brain_mgz_raw,args.input+'/mri/raw_brain.mgz')
nib.save(brain_mgz,args.input+'/mri/norm.mgz')
