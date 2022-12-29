import os
from typing import Union, Optional, Tuple

import pandas as pd
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from pytorch_msssim import SSIM, ssim
from torchmetrics.functional import peak_signal_noise_ratio, mean_squared_error

from model.W3DIP import W3DIP, l2_regularization, W3DIPMultiPatch
from utils.common_utils import count_parameters, report_memory_usage, store_volume_nii_gz
from train.deconv_utils import shifter_kernel


class W3DIPMultiPatchTrainer:
    def __init__(
            self,
            w3dip: W3DIPMultiPatch,
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
            'image_name': [],
            'step': [],
            'ssim': [],
            'psnr': [],
            'mse': []
        }

        self.kernel_estimate_metrics = {
            'step': [],
            'mse': []
        }

        self.baseline_blurr_to_gt_metrics = {
            'image_name': [],
            'ssim': [],
            'psnr': [],
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
            [{'params': image_generator.parameters()} for image_generator in w3dip.image_generators] +
            [{'params': w3dip.kernel_gen.parameters(), 'lr': self.lr_kernel_network}],
            lr=self.lr_img_network
        )
        self.scheduler = MultiStepLR(self.optimizer, **lr_schedule_params)  # learning rates

    def fit_no_guidance(
            self,
            blurred_volumes: torch.FloatTensor,
            blurred_volume_names: Tuple[str, ...],
            sharp_volumes_ground_truth: Optional[torch.FloatTensor] = None,
            kernel_ground_truth: Optional[torch.FloatTensor] = None,
            kernel_ground_truth_name: Optional[str] = None,
            num_steps: int = 5000,
            mse_to_ssim_step: int = 1000,
            checkpoint_schedule: tuple = ([50, 25], [250, 100], [1000, 250], [2000, 500]),
            checkpoint_base_dir: str = os.path.join('..', 'results'),
            check_memory_usage: bool = False,
            checkpoint_volumes = False
    ):

        # Report memory usage
        report_memory_usage(things_in_gpu="Model", print_anyways=True)

        # Report model summary
        print('Kernel Generating Network')
        count_parameters(self.w3dip.kernel_gen)

        print('Image Generating Network')
        count_parameters(self.w3dip.image_generators[0])

        # Initialization for outer-loop
        save_frequency_schedule_loop = list(checkpoint_schedule)
        save_freq_change, save_freq = save_frequency_schedule_loop.pop(0)

        # Record baseline metrics
        if sharp_volumes_ground_truth is not None and sharp_volumes_ground_truth.shape[0] > 0:

            for blurred_vol_name, blurred_vol, sharp_vol_gt \
                    in zip(blurred_volume_names, blurred_volumes, sharp_volumes_ground_truth):
                self._record_baseline_metrics(blurred_vol_name, blurred_vol, sharp_vol_gt)

        # Save the kernel that will be used for blurring
        if kernel_ground_truth is not None:
            self.checkpoint_kernel_ground_truth(checkpoint_base_dir, kernel_ground_truth, kernel_ground_truth_name)

        # Save the blurred volumes as well
        self.checkpoint_input_blurred_volumes(blurred_volume_names, blurred_volumes, checkpoint_base_dir)

        # Start training loop
        for step in tqdm(range(num_steps)):
            # Forward pass
            sharp_vol_estimates, blur_kernel_estimate, blurr_vol_estimates = self.w3dip()

            if check_memory_usage:
                report_memory_usage(things_in_gpu="Model and Maps")

            # Measure loss
            l2_reg = self.w_k * l2_regularization(blur_kernel_estimate)

            # Average data fitting terms of all patches
            data_fitting_term = self.mse(blurr_vol_estimates, blurred_volumes) if step < mse_to_ssim_step else \
                1 - self.ssim(blurr_vol_estimates, blurred_volumes)

            data_fitting_term = data_fitting_term / len(blurr_vol_estimates)

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

            # Save progress of each patch
            if step % save_freq == 0 or step == num_steps - 1:
                self.checkpoint_net_level_metrics(
                    step=step,
                    kernel_estimate=blur_kernel_estimate,
                    base_output_dir=checkpoint_base_dir,
                    patch_filename=f'step_{step}',
                    kernel_ground_truth=kernel_ground_truth,
                )

                for blur_vol_est, sharp_vol_est, blurred_vol_name, blurred_vol, sharp_vol_gt \
                        in zip(blurr_vol_estimates, sharp_vol_estimates, blurred_volume_names,
                               blurred_volumes, sharp_volumes_ground_truth):

                    self.checkpoint_vol_level_outputs_and_metrics(
                        step=step,
                        blurred_vol_estimate=blur_vol_est,
                        sharpened_vol_estimate=sharp_vol_est,
                        base_output_dir=checkpoint_base_dir,
                        patch_filename=f'step_{step}',
                        sharp_volume_ground_truth=sharp_vol_gt,
                        blurred_vol_name=blurred_vol_name,
                        checkpoint_volumes=checkpoint_volumes
                    )

        # Clean up
        del sharp_vol_estimates
        del blur_kernel_estimate
        del blurr_vol_estimates
        torch.cuda.empty_cache()

    def checkpoint_input_blurred_volumes(self, blurred_volume_names, blurred_volumes, checkpoint_base_dir):
        for blurred_volume_name, blurred_volume in zip(blurred_volume_names, blurred_volumes):
            volume_dir = os.path.join(checkpoint_base_dir, blurred_volume_name.strip('.nii.gz'))
            os.makedirs(volume_dir, exist_ok=True)
            store_volume_nii_gz(
                vol_array=blurred_volume[0].cpu().detach().numpy(),
                volume_filename=blurred_volume_name,
                output_dir=volume_dir)

    def checkpoint_kernel_ground_truth(self, checkpoint_base_dir, kernel_ground_truth, kernel_ground_truth_name):
        blurr_kernel_np = kernel_ground_truth[0].cpu().detach().numpy().copy()
        blurr_kernel_np /= np.max(blurr_kernel_np)
        # Create dir where the results will be stored
        kernel_estimate_dir = os.path.join(checkpoint_base_dir, 'kernel_estimate')
        os.makedirs(kernel_estimate_dir, exist_ok=True)
        store_volume_nii_gz(
            vol_array=blurr_kernel_np,
            volume_filename=f"{kernel_ground_truth_name}.nii.gz",
            output_dir=kernel_estimate_dir
        )
        store_volume_nii_gz(
            vol_array=(blurr_kernel_np > 0.1).astype(np.uint8),
            volume_filename=f"{kernel_ground_truth_name}_seg.nii.gz",
            output_dir=kernel_estimate_dir
        )

    def _record_baseline_metrics(self, blurred_volume_name, blurred_volume, sharp_volume_ground_truth):
        # Add image name for which we are calculating the baseline metrics
        self.baseline_blurr_to_gt_metrics['image_name'].append(blurred_volume_name)

        # Calculate SSIM
        self.baseline_blurr_to_gt_metrics['ssim'].append(
            ssim(blurred_volume, sharp_volume_ground_truth).item())

        # Calculate PSNR
        self.baseline_blurr_to_gt_metrics['psnr'].append(
            peak_signal_noise_ratio(blurred_volume, sharp_volume_ground_truth).item())

        # Calculate MSE
        self.baseline_blurr_to_gt_metrics['mse'].append(
            mean_squared_error(blurred_volume, sharp_volume_ground_truth).item())

    def _record_sharp_vol_estimation_metrics(self, sharp_volume_estimate, sharp_volume_ground_truth, step, vol_name):
        self.image_estimate_metrics['image_name'].append(vol_name)

        self.image_estimate_metrics['step'].append(step)

        # Calculate SSIM
        self.image_estimate_metrics['ssim'].append(
            ssim(sharp_volume_estimate, sharp_volume_ground_truth).item())

        # Calculate PSNR
        self.image_estimate_metrics['psnr'].append(
            peak_signal_noise_ratio(sharp_volume_estimate, sharp_volume_ground_truth).item())

        # Calculate MSE
        self.image_estimate_metrics['mse'].append(
            mean_squared_error(sharp_volume_estimate, sharp_volume_ground_truth).item())

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
            vol_array=kernel_ground_truth_max_overlap[0],
            volume_filename=f"kernel_ground_truth_max_overlap.nii.gz",
            output_dir=step_output_dir)

        store_volume_nii_gz(
            vol_array=(kernel_ground_truth_max_overlap[0] > 0.1).astype(np.uint8),
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
        pd.DataFrame(self.loss_history).to_csv(os.path.join(output_dir, 'loss_history.csv'), index=False)

    def checkpoint_sharp_volume_estimation_metrics(self, output_dir: str, vol_name: str):
        pd.DataFrame(self.image_estimate_metrics).set_index('image_name').sort_index().loc[vol_name].to_csv(
            os.path.join(output_dir, 'sharp_volume_estimate_metrics.csv'), index=False)

    def checkpoint_kernel_estimation_metrics(self, output_dir: str):
        pd.DataFrame(self.kernel_estimate_metrics).to_csv(
            os.path.join(output_dir, 'kernel_estimate_metrics.csv'), index=False)

    def checkpoint_bare_baseline_metrics(self, output_dir: str):
        pd.DataFrame(self.baseline_blurr_to_gt_metrics).to_csv(
            os.path.join(output_dir, 'baseline_blurr_to_gt_metrics.csv'), index=False)

    def checkpoint_net_level_metrics(
            self,
            step: int,
            kernel_estimate: Union[torch.cuda.FloatTensor, torch.Tensor],
            base_output_dir: str, patch_filename: str,
            kernel_ground_truth: Optional[torch.FloatTensor] = None,
    ):
        self.checkpoint_loss(base_output_dir)

        self.checkpoint_bare_baseline_metrics(output_dir=base_output_dir)

        step_output_dir = os.path.join(base_output_dir, 'kernel_estimate', f'step_{step}')
        os.makedirs(step_output_dir, exist_ok=True)

        if kernel_ground_truth is not None:
            self._record_kernel_estimation_error_metrics(kernel_estimate, kernel_ground_truth, step, step_output_dir)
            self.checkpoint_kernel_estimation_metrics(output_dir=base_output_dir)

        kernel_estimate = kernel_estimate.cpu().detach().numpy().copy()
        kernel_estimate /= np.max(kernel_estimate)
        store_volume_nii_gz(
            vol_array=kernel_estimate[0, 0],
            volume_filename=f"kernel_estimate__{patch_filename}.nii.gz",
            output_dir=step_output_dir)

        store_volume_nii_gz(
            vol_array=(kernel_estimate[0, 0] > 0.1).astype(np.uint8),
            volume_filename=f"kernel_estimate_seg_mask_0.1_threshold__{patch_filename}.nii.gz",
            output_dir=step_output_dir)

    def checkpoint_vol_level_outputs_and_metrics(
            self,
            step: int,
            blurred_vol_estimate: Union[torch.cuda.FloatTensor, torch.Tensor],
            sharpened_vol_estimate: Union[torch.cuda.FloatTensor, torch.Tensor],
            blurred_vol_name: str,
            base_output_dir: str, patch_filename: str,
            sharp_volume_ground_truth: Optional[torch.FloatTensor] = None,
            checkpoint_volumes = False
    ):
        vol_output_dir = os.path.join(base_output_dir, blurred_vol_name.strip('.nii.gz'))
        os.makedirs(vol_output_dir, exist_ok=True)

        step_output_dir = os.path.join(vol_output_dir, f'step_{step}')
        os.makedirs(step_output_dir, exist_ok=True)

        # Record metrics
        if sharp_volume_ground_truth is not None:
            self._record_sharp_vol_estimation_metrics(
                sharpened_vol_estimate, sharp_volume_ground_truth, step, blurred_vol_name)
            self.checkpoint_sharp_volume_estimation_metrics(vol_name=blurred_vol_name, output_dir=vol_output_dir)

        if checkpoint_volumes:
            # Store outputs
            store_volume_nii_gz(
                vol_array=blurred_vol_estimate[0].cpu().detach().numpy(),
                volume_filename=f"blurred_vol_estimate__{patch_filename}.nii.gz",
                output_dir=step_output_dir)

            store_volume_nii_gz(
                vol_array=sharpened_vol_estimate[0].cpu().detach().numpy(),
                volume_filename=f"sharpened_vol_estimate__{patch_filename}.nii.gz",
                output_dir=step_output_dir)









