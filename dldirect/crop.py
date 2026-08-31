import nibabel as nib
import numpy as np
import argparse


def get_crop(volume):
    nonempty = np.argwhere(volume)
    top_left = nonempty.min(axis=0)
    bottom_right = nonempty.max(axis=0)
    
    return (top_left, bottom_right)


def apply_crop(volume, crop):
    top_left, bottom_right = crop
    cropped = volume[top_left[0]:bottom_right[0]+1,
                   top_left[1]:bottom_right[1]+1,
                   top_left[2]:bottom_right[2]+1]
    return cropped


def apply_uncrop(template, volume, crop):
    uncropped = np.zeros_like(template).astype(volume.dtype)
    top_left, bottom_right = crop
    
    uncropped[top_left[0]:bottom_right[0]+1,
                   top_left[1]:bottom_right[1]+1,
                   top_left[2]:bottom_right[2]+1] = volume
    return uncropped


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop/uncrop nifti to bounding-box of given template')

    parser.add_argument(
        '--revert',
        type=bool,
        required=False,
        default=False,
        help='Revert cropping'
    )

    parser.add_argument(
        'template',
        type=str,
        help='Use bounding-box from template'
    )
    
    parser.add_argument(
        'source',
        type=str,
        help='Input volume to crop'
    )
    
    parser.add_argument(
        'destination',
        type=str,
        help='Cropped output volume'
    )

    args = parser.parse_args()
    
    template_nib = nib.load(args.template)
    src_nib = nib.load(args.source)
    crop = get_crop(np.asanyarray(template_nib.dataobj))
    result = apply_crop(np.asanyarray(src_nib.dataobj), crop) if not args.revert else apply_uncrop(np.asanyarray(template_nib.dataobj), np.asanyarray(src_nib.dataobj), crop)

    # A cropped volume no longer starts at the origin of the template, so the
    # affine has to be shifted by the crop offset. Uncropping restores the
    # original array position and keeps the template affine unchanged.
    affine = template_nib.affine.copy()
    if not args.revert:
        affine[:3, 3] += template_nib.affine[:3, :3] @ crop[0]

    nib.save(nib.Nifti1Image(result, affine), args.destination)
    