import os
import warnings
from typing import Union, Tuple, List

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import nibabel as nib

from dataset.NIB_Dataset import NibDataset
from model.W3DIP import W3DIP
from utils.common_utils import count_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#warnings.filterwarnings("ignore")


def report_memory_usage(things_in_gpu: str, total_memory_threshold: float = 0.4):
    if 'cuda' in device.type:
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_gpu_memory = torch.cuda.memory_reserved(0)
        allocated_gpu_memory = torch.cuda.memory_allocated(0)

        if allocated_gpu_memory / total_gpu_memory > total_memory_threshold:
            print(f"Total memory available: {total_gpu_memory / 2**30: 0.3f}")
            print(f"Total memory reserved: {reserved_gpu_memory / 2 ** 30}")
            print(f"{things_in_gpu} Occupies: {allocated_gpu_memory/ 2**30} GB."
                  f"\n\t {torch.cuda.memory_allocated(0)/reserved_gpu_memory: 0.3f} % of reserved memory."
                  f"\n\t {torch.cuda.memory_allocated(0)/total_gpu_memory: 0.3f} % of total memory.")
    else:
        print("No GPU available")


def store_volume_nii_gz(tensor: Union[torch.cuda.FloatTensor, torch.Tensor], volume_filename: str, output_dir:str):
    nib_img = nib.Nifti1Image(
        tensor.cpu().detach().numpy().astype(np.uint16),
        np.eye(4)
    )
    nib.save(nib_img, os.path.join(output_dir, volume_filename))


def checkpoint_outputs(blurred_vol_estimate: Union[torch.cuda.FloatTensor, torch.Tensor],
                       sharpened_vol_estimate: Union[torch.cuda.FloatTensor, torch.Tensor],
                       kernel_estimate: Union[torch.cuda.FloatTensor, torch.Tensor],
                       output_dir: str, target_patch_filepath: str):

    step_output_dir = os.path.join(output_dir, f'step_{step}')
    os.makedirs(step_output_dir, exist_ok=True)
    patch_filename = os.path.basename(target_patch_filepath)

    store_volume_nii_gz(
        tensor=blurred_vol_estimate, volume_filename=f"blurred_vol_estimate__{patch_filename}",
        output_dir=step_output_dir)
    store_volume_nii_gz(
        tensor=sharpened_vol_estimate, volume_filename=f"sharpened_vol_estimate__{patch_filename}",
        output_dir=step_output_dir)
    store_volume_nii_gz(
        tensor=kernel_estimate, volume_filename=f"kernel_estimate__{patch_filename}",
        output_dir=step_output_dir)


if __name__ == '__main__':
    # General params
    LR = 0.01
    num_iter = 3000
    kernel_size_estimate = (5, 5, 10)
    save_frequency_schedule = [(50, 25), (250, 100), (1000, 250), (2000, 500)]
    output_dir = os.path.join('..', '..', 'results', 'test_runs', '64x64x128')
    os.makedirs(output_dir, exist_ok=True)

    # Load volume to fit
    blurred_patches_dir = os.path.join("..", "..", "data", "blurred_patches")
    blurred_patch_dir = os.path.join(blurred_patches_dir, "gaussian_sigmas_xyz_1.1_1.1_1.85_size_5_5_10")
    nib_dataset = NibDataset(input_volume_dir=blurred_patch_dir, dtype=np.float32)
    target_blurred_patch = nib_dataset.__getitem__(1).to(device)
    target_patch_filepath = nib_dataset.file_paths[1]
    print(target_blurred_patch.shape)

    # Only for debugging and understanding the mappings
    w3dip = W3DIP(
        img_gen_input_noise_spatial_size=tuple(target_blurred_patch.size()[1:]),
        img_gen_output_channels=target_blurred_patch.size()[0],
        img_gen_input_noise_num_channels=8,
        img_gen_upsample_strategy='transposed_conv',
        kernel_net_noise_input_size=200,
        kernel_net_num_hidden=1000,
        estimated_kernel_shape=kernel_size_estimate
    )
    w3dip.input_noises_to_cuda()
    w3dip.to(device)

    # Losses
    mse = torch.nn.MSELoss().to(device)

    # Report memory usage
    report_memory_usage(things_in_gpu="Model")

    # Report model summary
    count_parameters(w3dip)

    # Define Optimizer
    optimizer = torch.optim.Adam(
        [{'params': w3dip.image_gen.parameters()}, {'params': w3dip.kernel_gen.parameters(), 'lr': 1e-4}], lr=LR
    )
    scheduler = MultiStepLR(optimizer, milestones=[2000, 3000, 4000], gamma=0.5)  # learning rates

    loss_history = {
        'total': [],
        'data_fitting_term': [],
        'wiener_term': [],
        'gen_kernel_similarity_to_init_kernel': [],
        'l2_reg_kernel': []
    }

    # Initialization for outer-loop
    save_freq_change, save_freq = save_frequency_schedule.pop(0)
    for step in tqdm(range(num_iter)):
        # Forward pass
        out_x, out_k = w3dip()

        # Get blurred estimate
        out_k_m = out_k.view(-1, 1, *kernel_size_estimate)
        out_y = nn.functional.conv3d(out_x, out_k_m, padding='same', bias=None)

        # Measure loss
        L_MSE = mse(out_y.squeeze_(), target_blurred_patch.squeeze_())

        report_memory_usage(things_in_gpu="Model and Maps")

        # Backprop
        L_MSE.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Store loss values
        loss_history['total'].append(L_MSE.item())
        loss_history['data_fitting_term'].append(0)
        loss_history['wiener_term'].append(0)
        loss_history['gen_kernel_similarity_to_init_kernel'].append(0)
        loss_history['l2_reg_kernel'].append(0)

        # Save intermediate outputs
        if step > save_freq_change and len(save_frequency_schedule) > 0:
            save_freq_change, save_freq = save_frequency_schedule.pop(0)

        if step % save_freq == 0:
            if step == 198:
                print('hallo')
            checkpoint_outputs(
                blurred_vol_estimate=out_y, sharpened_vol_estimate=out_x.squeeze_(), kernel_estimate=out_k_m.squeeze_(),
                output_dir=output_dir, target_patch_filepath=target_patch_filepath
            )

            # Store loss history
            pd.DataFrame(loss_history).to_csv(os.path.join(output_dir, 'loss_history.csv'))

    # Clean up
    torch.cuda.empty_cache()

