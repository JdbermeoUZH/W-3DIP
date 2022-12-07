import os
from typing import Union, Optional

import pandas as pd
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from pytorch_msssim import SSIM, ssim
from torchmetrics.functional import peak_signal_noise_ratio, mean_squared_error

from model.W3DIP import W3DIP, l2_regularization
from utils.common_utils import count_parameters, report_memory_usage, store_volume_nii_gz
from train.deconv_utils import shifter_kernel


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
        self.ssim = SSIM(channel=1, spatial_dims=3).to(device)
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
            sharp_volume_ground_truth: Optional[torch.FloatTensor] = None,
            kernel_ground_truth: Optional[torch.FloatTensor] = None,
            num_steps: int = 5000,
            mse_to_ssim_step: int = 1000,
            checkpoint_schedule: tuple = ([50, 25], [250, 100], [1000, 250], [2000, 500]),
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

            data_fitting_term = self.mse(out_y, blurred_volume[None, ]) if step < mse_to_ssim_step else \
                1 - self.ssim(out_y, blurred_volume[None, ])

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
            if step > save_freq_change and len(save_frequency_schedule_loop) > 0:
                save_freq_change, save_freq = save_frequency_schedule_loop.pop(0)

            if step % save_freq == 0 or step == num_steps - 1:
                self.checkpoint_network_outputs_and_metrics(
                    step=step,
                    blurred_vol_estimate=out_y[0, 0],
                    sharpened_vol_estimate=out_x[0, 0],
                    kernel_estimate=out_k[0, 0],
                    output_dir=checkpoint_base_dir,
                    patch_filename=f'step_{step}',
                    sharp_volume_ground_truth=sharp_volume_ground_truth,
                    kernel_ground_truth=kernel_ground_truth
                )

        # Clean up
        del out_x
        del out_y
        del out_k
        torch.cuda.empty_cache()

    def _record_sharp_vol_estimation_metrics(self, sharp_volume_estimate, sharp_volume_ground_truth, step):
        self.image_estimate_metrics['step'].append(step)

        # Calculate SSIM
        self.image_estimate_metrics['ssim'].append(
            ssim(sharp_volume_estimate[None], sharp_volume_ground_truth).item())

        # Calculate PSNR
        self.image_estimate_metrics['psnr'].append(
            peak_signal_noise_ratio(sharp_volume_estimate[None], sharp_volume_ground_truth).item())

        # Calculate MSE
        self.image_estimate_metrics['mse'].append(
            mean_squared_error(sharp_volume_estimate[None], sharp_volume_ground_truth).item())

    def _record_kernel_estimation_error_metrics(self, kernel_estimate, kernel_ground_truth, step, step_output_dir):
        # Find most likely translation by choosing the one with the lowest MSE
        search_window = tuple(((np.array(kernel_estimate.shape[-3:]) - 1) / 2).astype(int))
        min_mse, displacement_highest_overlap, kernel_ground_truth_max_overlap = \
            shifter_kernel(mover=kernel_ground_truth, target=kernel_estimate, search_window=search_window)

        # Add metrics to dictionary
        self.kernel_estimate_metrics['step'].append(step)
        self.kernel_estimate_metrics['mse'].append(min_mse.item())

        # Store the version or portion of the ground truth kernel with the highest overlap
        kernel_ground_truth_max_overlap = kernel_ground_truth_max_overlap.cpu().detach().numpy().copy()
        kernel_ground_truth_max_overlap /= np.max(kernel_ground_truth_max_overlap)

        store_volume_nii_gz(
            vol_array=kernel_ground_truth_max_overlap,
            volume_filename=f"kernel_ground_truth_max_overlap.nii.gz",
            output_dir=step_output_dir)

        store_volume_nii_gz(
            vol_array=(kernel_ground_truth_max_overlap > 0.1).astype(np.uint8),
            volume_filename=f"kernel_ground_truth_max_overlap_seg_mask_0.1_threshold.nii.gz",
            output_dir=step_output_dir)

    def _record_losses(self, total: float, data_fitting_term: float,
                       wiener_term: Optional[float] = None,
                       estim_to_init_kernel_mse: Optional[float] = None,
                       l2_reg_kernel: Optional[float] = None
                       ):

        self.loss_history['total'].append(total)
        self.loss_history['data_fitting_term'].append(data_fitting_term)
        self.loss_history['wiener_term'].append(0 if wiener_term is None else wiener_term)
        self.loss_history['estim_to_init_kernel_mse'].append(0 if estim_to_init_kernel_mse is None
                                                             else estim_to_init_kernel_mse)
        self.loss_history['l2_reg_kernel'].append(0 if l2_reg_kernel is None else l2_reg_kernel)

    def checkpoint_loss(self, output_dir: str):
        # Store loss history
        pd.DataFrame(self.loss_history).to_csv(os.path.join(output_dir, 'loss_history.csv'))

    def checkpoint_sharp_volume_estimation_metrics(self, output_dir: str):
        pd.DataFrame(self.image_estimate_metrics).to_csv(
            os.path.join(output_dir, 'sharp_volume_estimate_metrics.csv'))

    def checkpoint_kernel_estimation_metrics(self, output_dir: str):
        pd.DataFrame(self.kernel_estimate_metrics).to_csv(
            os.path.join(output_dir, 'kernel_estimate_metrics.csv'))

    def checkpoint_network_outputs_and_metrics(
            self,
            step: int,
            blurred_vol_estimate: Union[torch.cuda.FloatTensor, torch.Tensor],
            sharpened_vol_estimate: Union[torch.cuda.FloatTensor, torch.Tensor],
            kernel_estimate: Union[torch.cuda.FloatTensor, torch.Tensor],
            output_dir: str, patch_filename: str,
            sharp_volume_ground_truth: Optional[torch.FloatTensor] = None,
            kernel_ground_truth: Optional[torch.FloatTensor] = None
    ):
        step_output_dir = os.path.join(output_dir, f'step_{step}')
        os.makedirs(step_output_dir, exist_ok=True)

        # Record metrics
        self.checkpoint_loss(output_dir=output_dir)

        if sharp_volume_ground_truth is not None:
            self._record_sharp_vol_estimation_metrics(sharpened_vol_estimate, sharp_volume_ground_truth, step)
            self.checkpoint_sharp_volume_estimation_metrics(output_dir=output_dir)

        if kernel_ground_truth is not None:
            self._record_kernel_estimation_error_metrics(kernel_estimate, kernel_ground_truth, step, step_output_dir)
            self.checkpoint_kernel_estimation_metrics(output_dir=output_dir)

        # Store outputs
        store_volume_nii_gz(
            vol_array=blurred_vol_estimate.cpu().detach().numpy(),
            volume_filename=f"blurred_vol_estimate__{patch_filename}.nii.gz",
            output_dir=step_output_dir)

        store_volume_nii_gz(
            vol_array=sharpened_vol_estimate.cpu().detach().numpy(),
            volume_filename=f"sharpened_vol_estimate__{patch_filename}.nii.gz",
            output_dir=step_output_dir)

        kernel_estimate = kernel_estimate.cpu().detach().numpy().copy()
        kernel_estimate /= np.max(kernel_estimate)
        store_volume_nii_gz(
            vol_array=kernel_estimate,
            volume_filename=f"kernel_estimate__{patch_filename}.nii.gz",
            output_dir=step_output_dir)

        store_volume_nii_gz(
            vol_array=(kernel_estimate > 0.1).astype(np.uint8),
            volume_filename=f"kernel_estimate_seg_mask_0.1_threshold__{patch_filename}.nii.gz",
            output_dir=step_output_dir)








