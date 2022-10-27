import os
import glob
from typing import Type, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.common_utils import np_to_torch


class NibDataset(Dataset):
    def __init__(
            self,
            input_volume_dir: str,
            transform=None,
            device: str = None,
            dtype: Type[np.dtype] = np.float32
    ):
        self.input_volume_dir = input_volume_dir
        self.transform = transform
        self.file_paths = glob.glob(os.path.join(self.input_volume_dir, "*.nii.gz"))
        self.dtype = dtype

        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        img = nib.load(self.file_paths[idx])  # !Image.open(os.path.join(self.root_dir,img_name))

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
                        image_name = os.path.basename(self.file_paths[idx]).split('.nii.gz')[0]
                        patch_filename = f'vol_{image_name}_pos_x_{patch_center_x}' \
                                         f'_pos_y_{patch_center_y}_pos_z_{patch_center_z}.nii.gz'
                        nib.save(nib_img, os.path.join(patch_persist_dir, patch_filename))

        # Restack the patches along a single dim and return them
        return patches.reshape(-1, *patch_size_xyz)


if __name__ == "__main__":
    simulated_vessels_dir = os.path.join("..", "..", "..", "data", "raw_data", "simulated_vessels")
    raw_data_dir = os.path.join(simulated_vessels_dir, "raw")
    nib_dataset = NibDataset(input_volume_dir=raw_data_dir, dtype=np.float32)
    print(nib_dataset[0])

    nib_dataset.sample_patches_from_volume(
        idx=0,
        patch_size_xyz=(300, 300, 30),
        num_patches_xyz=(1, 1, 50),
        patch_persist_dir=os.path.join(simulated_vessels_dir, "ground_truth_patches", "300x300x30")
    )


