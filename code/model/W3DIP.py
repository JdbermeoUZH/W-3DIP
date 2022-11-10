import os
import warnings
from typing import Union, Tuple, List

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from dataset.NIB_Dataset import NibDataset
from model.KernelGenerator import KernelGenerator
from model.ImageGenerator import ImageGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#warnings.filterwarnings("ignore")


class W3DIP(nn.Module):
    def __init__(
            self,
            img_gen_input_noise_spatial_size: Union[int, Tuple[int, ...]],
            img_gen_output_channels: int = 1,
            img_gen_input_noise_num_channels: int = 8,
            img_gen_input_noise_reg_noise_std: float = 0.001,
            img_gen_upsample_strategy: str = 'bilinear',
            kernel_net_noise_input_size: int = 200,
            kernel_net_num_hidden: int = 1000,
            estimated_kernel_shape: Tuple[int, ...] = (5, 5, 10),
    ):
        super(W3DIP, self).__init__()
        self.image_gen = ImageGenerator(
            num_output_channels=img_gen_output_channels,
            input_noise_spatial_size=img_gen_input_noise_spatial_size,
            input_noise_num_channels=img_gen_input_noise_num_channels,
            input_noise_reg_noise_std=img_gen_input_noise_reg_noise_std,
            upsample_strategy=img_gen_upsample_strategy
        )

        self.kernel_gen = KernelGenerator(
            noise_input_size=kernel_net_noise_input_size, num_hidden=kernel_net_num_hidden,
            estimated_kernel_shape=estimated_kernel_shape
        )

    def forward(self):
        return self.image_gen(), self.kernel_gen()


if __name__ == '__main__':
    # General params
    LR = 0.01
    num_iter = 100
    kernel_size_estimate = (5, 5, 10)
    save_frequency_schedule = [(50, 5), (1000, 100), (1000, 250)]

    # Load volume to fit
    blurred_patches_dir = os.path.join("..", "..", "..", "data", "blurred_patches")
    blurred_patch_dir = os.path.join(blurred_patches_dir, "gaussian_sigmas_xyz_1.1_1.1_1.85_size_5_5_10")
    nib_dataset = NibDataset(input_volume_dir=blurred_patch_dir, dtype=np.float32)
    blurred_patch = nib_dataset.__getitem__(1)
    print(blurred_patch.shape)

    # Only for debugging and understanding the mappings
    w3dip = W3DIP(
        img_gen_input_noise_spatial_size=tuple(blurred_patch.size()[1:]),
        img_gen_output_channels=blurred_patch.size()[0],
        img_gen_input_noise_num_channels=8,
        img_gen_upsample_strategy='transposed_conv',
        kernel_net_noise_input_size=200,
        kernel_net_num_hidden=1000,
        estimated_kernel_shape=kernel_size_estimate
    )
    w3dip.to(device)

    # Losses
    mse = torch.nn.MSELoss().to(device)

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
    # TODO: Create basic training loop
    save_freq_change, save_freq = save_frequency_schedule.pop(0)
    for step in tqdm(range(num_iter)):
        # Forward pass
        out_x, out_k = w3dip()

        # Get blurred estimate
        out_k_m = out_k.view(-1, 1, *kernel_size_estimate)
        out_y = nn.functional.conv3d(out_x, out_k_m, padding='same', bias=None)

        # Measure loss
        L_MSE = mse(out_y, blurred_patch)

        # Backprop
        optimizer.step()
        scheduler.step(step)
        optimizer.zero_grad()
        #conv3d(patch.float(), torch.unsqueeze(kernel, dim=0).float(), padding='same')

        # Save intermediate outputs
        if step > save_freq_change:
            save_freq_change, save_freq = save_frequency_schedule.pop(0)

        if step % save_freq == 0:
            print('store_img')
