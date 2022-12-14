import os
import glob
from typing import Type, Tuple, List, Union

import numpy as np
import nibabel as nib
import torch
from torch.nn.functional import conv3d

from dataset.NIB_Dataset import NibDataset
from utils.common_utils import np_to_torch, store_volume_nii_gz


def _noiseless_blurring(volume: torch.FloatTensor, kernel: torch.FloatTensor) -> torch.FloatTensor:
    return conv3d(volume, kernel, padding='same').float()

def _poisson_noise_blurring(volume: torch.FloatTensor, kernel: torch.FloatTensor) -> torch.FloatTensor:
    raise RuntimeError('Not implemented yet')


class SimulatedBlurDatasetSinglePatch(NibDataset):
    def __init__(
            self,
            input_volume_dir: str,
            kernels_dir: str,
            transform=None,
            device: Union[str, torch.device] = None,
            dtype: Type[np.dtype] = np.float32,
            noiseless_blurring: bool = True
    ):
        super(SimulatedBlurDatasetSinglePatch, self).__init__(input_volume_dir, transform, device, dtype)
        self.kernels_filepaths = glob.glob(os.path.join(kernels_dir, "*.npy"))
        self.blurring_fn = _noiseless_blurring if noiseless_blurring else _poisson_noise_blurring

    def __getitem__(self, idx: int) -> Tuple[Tuple[str, torch.FloatTensor], List[Tuple[str, torch.FloatTensor, torch.FloatTensor]]]:
        # Get ground_truth volume
        ground_truth_vol_name = os.path.basename(self.input_volume_filepaths[idx]).strip('.nii.gz')
        ground_truth_vol = nib.load(self.input_volume_filepaths[idx])

        # Convert image to PyTorch tensor
        ground_truth_vol = np_to_torch(ground_truth_vol.get_fdata(dtype=self.dtype)).to(self.device)

        # Obtain blurred versions of the ground truth image
        blur_kernels = [(os.path.basename(kernel_fp).strip('.npy'), np_to_torch(np.load(kernel_fp)).to(self.device))
                        for kernel_fp in self.kernels_filepaths]

        blurred_volumes = []

        for blur_kernel_name, blur_kernel in blur_kernels:
            blurred_vol = self.blurring_fn(ground_truth_vol.float(), torch.unsqueeze(blur_kernel, dim=0).float())\
                .to(self.device)
            blurred_volumes.append((blur_kernel_name, blur_kernel, blurred_vol))

        return (ground_truth_vol_name, ground_truth_vol), blurred_volumes


class SimulatedBlurDatasetMultiPatch(NibDataset):
    def __init__(
            self,
            input_volume_dir: str,
            kernels_dir: str,
            transform=None,
            device: Union[str, torch.device] = None,
            dtype: Type[np.dtype] = np.float32,
            noiseless_blurring: bool = True
    ):
        super(SimulatedBlurDatasetMultiPatch, self).__init__(input_volume_dir, transform, device, dtype)
        self.kernels_filepaths = glob.glob(os.path.join(kernels_dir, "*.npy"))
        self.blurring_fn = _noiseless_blurring if noiseless_blurring else _poisson_noise_blurring

    def __len__(self):
        return len(self.kernels_filepaths)

    def __getitem__(self, idx: int) -> Tuple[Tuple[str, torch.FloatTensor], List[Tuple[str, torch.FloatTensor, torch.FloatTensor]]]:
        # Get the kernel to use
        kernel_file_path = self.kernels_filepaths[idx]
        kernel_name = os.path.basename(kernel_file_path).strip('.npy')
        blur_kernel = np_to_torch(np.load(kernel_file_path)).to(self.device)

        # Get ground_truth volumes
        ground_truth_vol_names = [os.path.basename(input_volume_filepath).strip('.nii.gz')
                                 for input_volume_filepath in self.input_volume_filepaths]


        ground_truth_volumes = torch.stack(
            [np_to_torch(nib.load(input_volume_filepath).get_fdata(dtype=self.dtype)).to(self.device)
             for input_volume_filepath in self.input_volume_filepaths])

        # Get blurred volumes
        blurred_volumes = self.blurring_fn(ground_truth_volumes.float(), torch.unsqueeze(blur_kernel, dim=0)
                                           .float()).to(self.device)

        return ground_truth_vol_names, ground_truth_volumes, kernel_name, blur_kernel, blurred_volumes


if __name__ == '__main__':
    exp_dir = os.path.join('..', '..', '..', 'experiments', 'noiseless_kernels', 'random_patches_same_volume')

    # Let's test if the loading works
    w3dip_dataset_single_patch = SimulatedBlurDatasetSinglePatch(
        input_volume_dir=os.path.join(exp_dir, 'volumes', '64x64x128'),
        kernels_dir=os.path.join(exp_dir, 'kernels'),
        noiseless_blurring=True
    )

    w3dip_dataset_multi_patch = SimulatedBlurDatasetMultiPatch(
        input_volume_dir=os.path.join(exp_dir, 'volumes', '64x64x128'),
        kernels_dir=os.path.join(exp_dir, 'kernels'),
        noiseless_blurring=True
    )

    for ground_truth_volume_names, ground_truth_volumes, kernel_name, blur_kernel, blurred_volumes in w3dip_dataset_multi_patch:
        print(ground_truth_volume_names)
        print(ground_truth_volumes.shape)
        print(kernel_name)
        print(blur_kernel)
        print(blurred_volumes.shape)

    for (ground_truth_volume_name, ground_truth_volume), blurred_volumes in w3dip_dataset_single_patch:
        for kernel_name, kernel, blurred_image in blurred_volumes:
            print(ground_truth_volume_name)
            print(kernel_name)
            print(f'ground_truth_volume.shape: {ground_truth_volume.shape}')
            print(f'kernel.shape: {kernel.shape}')
            print(f'blurred_image.shape: {blurred_image.shape}')

            kernel = kernel.squeeze_().cpu().detach().numpy()
            kernel /= np.max(kernel)

            os.makedirs('./sanity_check_plots', exist_ok=True)
            # Save the volumes for a sanity check
            store_volume_nii_gz(
                vol_array=ground_truth_volume.squeeze_().cpu().detach().numpy(),
                volume_filename=f'ground_truth_{ground_truth_volume_name}.nii.gz',
                output_dir='./sanity_check_plots'
            )

            store_volume_nii_gz(
                vol_array=kernel,
                volume_filename=f'{kernel_name}.nii.gz',
                output_dir='./sanity_check_plots'
            )

            store_volume_nii_gz(
                vol_array=(kernel > 0.1).astype(np.uint8),
                volume_filename=f'{kernel_name}_seg.nii.gz',
                output_dir='./sanity_check_plots'
            )

            store_volume_nii_gz(
                vol_array=blurred_image.squeeze_().cpu().detach().numpy(),
                volume_filename=f'blurred_{kernel_name}_{ground_truth_volume_name}.nii.gz',
                output_dir='./sanity_check_plots'
            )

        break
