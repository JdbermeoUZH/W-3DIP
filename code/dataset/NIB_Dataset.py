import os
import glob

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
            dtype: np.dtype = np.float32
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

    def __getitem__(self, idx):
        img = nib.load(self.file_paths[idx])  # !Image.open(os.path.join(self.root_dir,img_name))

        # Convert image to PyTorch tensor
        img = np_to_torch(img.get_fdata(dtype=self.dtype))

        if self.transform:
            img = self.transform(img)
            return img
        else:
            return img


if __name__ == "__main__":
    raw_data_dir = os.path.join("..", "..", "..", "data", "raw_data", "simulated_vessels", "raw")
    nib_dataset = NibDataset(input_volume_dir=raw_data_dir, dtype=np.float32)
    print(nib_dataset[0])


