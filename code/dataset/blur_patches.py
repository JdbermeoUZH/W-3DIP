import os
import glob
import argparse

import torch
import nibabel as nib
from torch.nn.functional import conv3d
import numpy as np

from dataset.NIB_Dataset import NibDataset
from utils.common_utils import np_to_torch

torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--patches_dir', type=str)
    parser.add_argument('--kernels_dir', type=str)
    parser.add_argument('--blurred_patches_dir', type=str, default='./blurred_patches')
    args = parser.parse_args()

    # Get list of paths to each of the kernels
    kernel_npy_paths = glob.glob(os.path.join(args.kernels_dir, '*.npy'))

    # Apply each blurr kernel to each of the selected patches in patches_dir in the
    nib_dataset = NibDataset(input_volume_dir=args.patches_dir, dtype=np.float32)
    for idx, patch in enumerate(nib_dataset):
        for kernel_npy_path in kernel_npy_paths:
            # Load kernel
            kernel = np_to_torch(np.load(kernel_npy_path))
            blurred_vol = conv3d(patch.float(), torch.unsqueeze(kernel, dim=0).float(),  padding='same')

            # Store the blurred volume
            blurred_patch_dir = os.path.join(args.blurred_patches_dir, os.path.basename(kernel_npy_path).split('.npy')[0])
            os.makedirs(blurred_patch_dir, exist_ok=True)

            nib_img = nib.Nifti1Image(
                torch.squeeze(blurred_vol).numpy().astype(np.uint16),
                np.eye(4)
            )

            nib.save(nib_img,
                     os.path.join(blurred_patch_dir, os.path.basename(nib_dataset.input_volume_filepaths[idx])))



