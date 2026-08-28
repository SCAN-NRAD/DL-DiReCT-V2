import argparse
import numpy as np
import nibabel as nib
import pandas as pd
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
args = parser.parse_args()

# Read and conform (256x256x256) DL+DiReCT results
# segmentation
seg_img = nib.load(args.input+'/softmax_seg.nii.gz')
seg_image = conform(seg_img, order=0, orientation = 'LIA')
# MRI 
brain_img = nib.load(args.input+'/T1w_norm_noskull_cropped.nii.gz')
brain_image = conform(brain_img, order=0, orientation = 'LIA')
affine = get_vox2ras_tkr(seg_image)

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

cc_slices = np.argwhere(output_data == label_ids['Corpus-Callosum']).T[0]
if cc_slices.size == 0:
    raise SystemExit('ERROR: no Corpus-Callosum was segmented, cannot separate the hemispheres.')

cc_min = int(cc_slices.min())
cc_max = int(cc_slices.max())

side = np.zeros_like(output_data)
side[0:cc_min+int((cc_max-cc_min)/2),:,:] = 10000
side[cc_min+int((cc_max-cc_min)/2):255,:,:] = 5000
midline_mask = np.isin(output_data, [label_ids[lbl] for lbl in midline_labels])
temp = np.where(midline_mask & (side == 10000), label_ids['Right-Cerebral-White-Matter'], output_data)
seg_img = np.where(midline_mask & (side == 5000), label_ids['Left-Cerebral-White-Matter'], temp)

# export the transformed segmentation
trans_seg_mgz = nib.freesurfer.mghformat.MGHImage(np.array(seg_img,dtype=np.int32) , affine, header=None, extra=None, file_map=None)
atlas_name = 'DKatlas' if np.max(seg_img) <= 3000 else '2009s'
nib.save(trans_seg_mgz,args.input+'/mri/aparc.'+atlas_name+'+aseg.mgz')
nib.save(trans_seg_mgz,args.input+'/mri/aparc.atlas+aseg.nii.gz')
nib.save(trans_seg_mgz,args.input+'/mri/aparc.atlas+aseg.mgz')

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
