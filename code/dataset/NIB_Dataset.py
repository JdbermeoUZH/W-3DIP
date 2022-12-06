import os
import glob
from typing import Type, Tuple, Union

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.common_utils import np_to_torch


def _create_segmentation_mask(
        kernel: np.ndarray,
        threshold_ranges: Tuple[int, ...] = (1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.25, 0.5)
):
    # Return a dictionary with the different segmentation masks that are non-empty
    seg_masks_dict = {}
    for threshold_range in threshold_ranges:
        seg_mask_i = (kernel > threshold_range).astype(np.uint16)
        if seg_mask_i.sum() > 0:
            seg_masks_dict[threshold_range] = seg_mask_i

    return seg_masks_dict


def create_segmentation_masks_kernel(kernel_nib_filepath: str):
    if os.path.exists(kernel_nib_filepath):
        kernel_array = nib.load(kernel_nib_filepath)

        kernel_segmentations = _create_segmentation_mask(kernel_array)

        dir_to_store = os.path.dirname(kernel_nib_filepath)

        for threshold, seg_mask in kernel_segmentations.items():
            nib.save(
                seg_mask,
                os.path.join(dir_to_store, f'kernel_segmentation_mask_thres_{threshold}..nii.gz')
            )


class NibDataset(Dataset):
    def __init__(
            self,
            input_volume_dir: str,
            transform=None,
            device: Union[str, torch.device] = None,
            dtype: Type[np.dtype] = np.float32
    ):
        self.input_volume_dir = input_volume_dir
        self.transform = transform
        self.input_volume_filepaths = glob.glob(os.path.join(self.input_volume_dir, "*.nii.gz"))
        self.dtype = dtype

        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.input_volume_filepaths)

    def __getitem__(self, idx: int) -> torch.FloatTensor:
        img = nib.load(self.input_volume_filepaths[idx])  # !Image.open(os.path.join(self.root_dir,img_name))

        # Convert image to PyTorch tensor
        img = np_to_torch(img.get_fdata(dtype=self.dtype))

        if self.transform:
            img = self.transform(img)
            return img
        else:
            return img

    def sample_patches_from_volume(self,
                                   idx: int,
                                   patch_size_xyz: Tuple[int, int, int],
                                   num_patches_xyz: Tuple[int, int, int],
                                   patch_persist_dir: str = None
                                   ):
        volume = self[idx]
        dims = volume.shape[-3:]
        strides = np.ceil(dims / np.array(num_patches_xyz)).astype(int)

        # Extract patches from the volume
        patches = volume\
            .unfold(-3, patch_size_xyz[0], strides[0])\
            .unfold(-3, patch_size_xyz[1], strides[1])\
            .unfold(-3, patch_size_xyz[2], strides[2])\
            .squeeze(0)

        if patch_persist_dir:
            os.makedirs(patch_persist_dir, exist_ok=True)
            for patch_num_x in range(patches.shape[0]):
                patch_center_x = patch_num_x * strides[0] + 1

                for patch_num_y in range(patches.shape[1]):
                    patch_center_y = patch_num_y * strides[1] + 1

                    for patch_num_z in range(patches.shape[2]):
                        patch_center_z = patch_num_z * strides[2] + 1

                        nib_img = nib.Nifti1Image(
                            patches[patch_num_x, patch_num_y, patch_num_z].numpy().astype(np.uint16),
                            np.eye(4)
                        )
                        image_name = os.path.basename(self.input_volume_filepaths[idx]).split('.nii.gz')[0]
                        patch_filename = f'vol_{image_name}_pos_x_{patch_center_x}' \
                                         f'_pos_y_{patch_center_y}_pos_z_{patch_center_z}.nii.gz'
                        nib.save(nib_img, os.path.join(patch_persist_dir, patch_filename))

        # Restack the patches along a single dim and return them
        return patches.reshape(-1, *patch_size_xyz)


if __name__ == "__main__":
    simulated_vessels_dir = os.path.join("..", "..", "..", "data", "raw_data", "simulated_vessels")
    raw_data_dir = os.path.join(simulated_vessels_dir, "raw")
    nib_dataset = NibDataset(input_volume_dir=raw_data_dir, dtype=np.float32)

    patch_size = (32, 32, 128)
    num_patches_xyz = (10, 10, 5)

    output_path = os.path.join("..", "..", "..", "data", "ground_truth", "simulated_vessels")
    os.makedirs(output_path, exist_ok=True)
    nib_dataset.sample_patches_from_volume(
        idx=0,
        patch_size_xyz=patch_size,
        num_patches_xyz=num_patches_xyz,
        patch_persist_dir=os.path.join(output_path, "patches", "x".join([str(size) for size in patch_size]))
    )


