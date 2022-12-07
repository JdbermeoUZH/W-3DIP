import os
import yaml
import argparse
from typing import Union, Tuple, Optional

import pandas as pd
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from dataset.SimulatedBlurDataset import SimulatedBlurDataset
from model.InputNoise import InputNoise
from model.W3DIP import W3DIP, l2_regularization
from model.ImageGenerator import ImageGeneratorInterCNN3D
from model.KernelGenerator import KernelGenerator
from run import step
from utils.common_utils import count_parameters, report_memory_usage, store_volume_nii_gz
from utils.SSIM import SSIM3D


class W3DIPTrainer:
    def __init__(
            self,
            w3dip: W3DIP,
            device: torch.device,
            lr_img_network: float = 1e-2,
            lr_kernel_network: float = 1e-4,
            lr_schedule_params: dict[str: list, str: float] = {"milestones": [2000, 3000, 4000], "gamma": 0.5},
            w_k: float = 0.001

    ):
        self.w3dip = w3dip
        self.w3dip.to_device(device)

        self.loss_history = {
            'total': [],
            'data_fitting_term': [],
            'wiener_term': [],
            'estim_to_init_kernel_mse': [],
            'l2_reg_kernel': []
        }

        self.image_estimate_metrics = {
            'step': [],
            'ssim': [],
            'mmssim': [],
            'psnr': [],
            'mse': []
        }

        self.kernel_estimate_metrics = {
            'step': [],
            'mse': []
        }

        self.device = device

        # Losses
        self.mse = torch.nn.MSELoss().to(device)
        self.ssim = SSIM3D().to(device)
        self.w_k = w_k

        # Optimizer
        self.lr_img_network = lr_img_network
        self.lr_kernel_network = lr_kernel_network

        self.optimizer = torch.optim.Adam(
            [{'params': w3dip.image_gen.parameters()},
             {'params': w3dip.kernel_gen.parameters(), 'lr': self.lr_kernel_network}],
            lr=self.lr_img_network
        )
        self.scheduler = MultiStepLR(self.optimizer, **lr_schedule_params)  # learning rates

    def fit_no_guidance(
            self,
            blurred_volume: torch.FloatTensor,
            num_steps: int = 5000,
            mse_to_ssim_step: int = 1000,
            checkpoint_schedule: list = [[50, 25], [250, 100], [1000, 250], [2000, 500]],
            checkpoint_base_dir: str = os.path.join('..', 'results'),
            check_memory_usage: bool = False
    ):

        # Report memory usage
        report_memory_usage(things_in_gpu="Model", print_anyways=True)

        # Report model summary
        count_parameters(self.w3dip)

        # Initialization for outer-loop
        save_frequency_schedule_loop = list(checkpoint_schedule)
        save_freq_change, save_freq = save_frequency_schedule_loop.pop(0)

        for step in tqdm(range(num_steps)):
            # Forward pass
            out_x, out_k, out_y = self.w3dip()

            if check_memory_usage:
                report_memory_usage(things_in_gpu="Model and Maps")

            # Measure loss
            l2_reg = self.w_k * l2_regularization(out_k)

            data_fitting_term = self.mse(out_y.squeeze_(), blurred_volume.squeeze_()) if step < mse_to_ssim_step else \
                1 - self.ssim(out_y, blurred_volume.reshape([1, 1] + list(blurred_volume.shape)))

            loss = l2_reg + data_fitting_term

            # Backprop
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # Store loss values
            self._record_losses(total=loss.item(), data_fitting_term=data_fitting_term.item(),
                                l2_reg_kernel=l2_reg.item())

            # Save intermediate outputs
            if step > checkpoint_schedule and len(checkpoint_schedule) > 0:
                save_freq_change, save_freq = checkpoint_schedule.pop(0)

            if step % save_freq == 0:
                self.checkpoint_outputs(
                    blurred_vol_estimate=out_y,
                    sharpened_vol_estimate=out_x.squeeze_(),
                    kernel_estimate=out_k.squeeze_(),
                    output_dir=checkpoint_base_dir,
                    patch_filename=f'step_{step}'
                )

        # Checkpoint after fitting
        self.checkpoint_outputs(
            blurred_vol_estimate=out_y,
            sharpened_vol_estimate=out_x.squeeze_(),
            kernel_estimate=out_k.squeeze_(),
            output_dir=checkpoint_base_dir,
            patch_filename=f'step_{step}'
        )

        # Clean up
        del out_x
        del out_y
        del out_k
        torch.cuda.empty_cache()

    def _record_losses(self, total: float, data_fitting_term: float,
                       wiener_term: Optional[float] = None,
                       estim_to_init_kernel_mse: Optional[float] = None,
                       l2_reg_kernel: Optional[float] = None
                       ):

        self.loss_history['total'].append(total)
        self.loss_history['data_fitting_term'].append(data_fitting_term)
        self.loss_history['wiener_term'].append(0 if wiener_term is None else wiener_term)
        self.loss_history['estim_to_init_kernel_mse'].append(0 if estim_to_init_kernel_mse is None else estim_to_init_kernel_mse)
        self.loss_history['l2_reg_kernel'].append(0 if l2_reg_kernel is None else l2_reg_kernel)

    def checkpoint_outputs(
            self,
            blurred_vol_estimate: Union[torch.cuda.FloatTensor, torch.Tensor],
            sharpened_vol_estimate: Union[torch.cuda.FloatTensor, torch.Tensor],
            kernel_estimate: Union[torch.cuda.FloatTensor, torch.Tensor],
            output_dir: str, patch_filename: str
    ):
        step_output_dir = os.path.join(output_dir, f'step_{step}')
        os.makedirs(step_output_dir, exist_ok=True)

        store_volume_nii_gz(
            vol_array=blurred_vol_estimate.cpu().detach().numpy(),
            volume_filename=f"blurred_vol_estimate__{patch_filename}.nii.gz",
            output_dir=step_output_dir)

        store_volume_nii_gz(
            vol_array=sharpened_vol_estimate.cpu().detach().numpy(),
            volume_filename=f"sharpened_vol_estimate__{patch_filename}.nii.gz",
            output_dir=step_output_dir)

        kernel_estimate = kernel_estimate.cpu().detach().numpy()
        kernel_estimate /= np.max(kernel_estimate)
        store_volume_nii_gz(
            vol_array=kernel_estimate,
            volume_filename=f"kernel_estimate__{patch_filename}.nii.gz",
            output_dir=step_output_dir)

        store_volume_nii_gz(
            vol_array=(kernel_estimate > 0.1).astype(np.uint8),
            volume_filename=f"kernel_estimate_seg_mask_0.1_threshold__{patch_filename}.nii.gz",
            output_dir=step_output_dir)

        # Store loss history
        pd.DataFrame(self.loss_history).to_csv(os.path.join(output_dir, 'loss_history.csv'))


if __name__ == '__main__':
    trainer = W3DIPTrainer()


