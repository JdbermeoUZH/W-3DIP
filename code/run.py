import os
import yaml
import argparse
from typing import Union, Tuple

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
from utils.common_utils import count_parameters, report_memory_usage, store_volume_nii_gz
from utils.SSIM import SSIM3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# warnings.filterwarnings("ignore")


def parse_arguments_and_load_config_file() -> Tuple[argparse.Namespace, dict]:
    parser = argparse.ArgumentParser(description='Subtask-2')
    parser.add_argument('--config_path_yaml', type=str,
                        help='Path to YAML configuration file overall benchmarking parameters')
    arguments = parser.parse_args()

    # Load parameters of configuration file
    with open(arguments.config_path_yaml, "r") as stream:
        try:
            yaml_config_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise exc

    return arguments, yaml_config_params


def checkpoint_outputs(blurred_vol_estimate: Union[torch.cuda.FloatTensor, torch.Tensor],
                       sharpened_vol_estimate: Union[torch.cuda.FloatTensor, torch.Tensor],
                       kernel_estimate: Union[torch.cuda.FloatTensor, torch.Tensor],
                       output_dir: str, patch_filename: str):
    step_output_dir = os.path.join(output_dir, f'step_{step}')
    os.makedirs(step_output_dir, exist_ok=True)

    store_volume_nii_gz(
        vol_array=blurred_vol_estimate.cpu().detach().numpy(),
        volume_filename=f"blurred_vol_estimate__{patch_filename}",
        output_dir=step_output_dir)

    store_volume_nii_gz(
        vol_array=sharpened_vol_estimate.cpu().detach().numpy(),
        volume_filename=f"sharpened_vol_estimate__{patch_filename}",
        output_dir=step_output_dir)

    kernel_estimate = kernel_estimate.cpu().detach().numpy()
    kernel_estimate /= np.max(kernel_estimate)
    store_volume_nii_gz(
        vol_array=kernel_estimate,
        volume_filename=f"kernel_estimate__{patch_filename}",
        output_dir=step_output_dir)

    store_volume_nii_gz(
        vol_array=(kernel_estimate > 0.1).astype(np.uint8),
        volume_filename=f"kernel_estimate_seg_mask_0.1_threshold__{patch_filename}",
        output_dir=step_output_dir)


def define_model_objects(
        target_patch_num_channels: int,
        target_patch_spatial_size: Tuple[int, ...],
        interCNN_feature_maps: Tuple[int, ...],
        kernel_gen_noise_input_size: int,
        kernel_gen_num_hidden: int,
        estimated_kernel_shape: Tuple[int, ...]
):
    w3dip = W3DIP(
        image_gen=ImageGeneratorInterCNN3D(
            num_output_channels=target_patch_num_channels,
            output_spatial_size=target_patch_spatial_size,
            input_noise=InputNoise(spatial_size=target_patch_spatial_size, num_channels=8, method='noise'),
            downsampling_output_channels=interCNN_feature_maps
        ),
        kernel_gen=KernelGenerator(
            noise_input_size=kernel_gen_noise_input_size,
            num_hidden=kernel_gen_num_hidden,
            estimated_kernel_shape=estimated_kernel_shape
        )
    )
    w3dip.input_noises_to_cuda()
    w3dip.to(device)

    # Report memory usage
    report_memory_usage(things_in_gpu="Model", print_anyways=True)

    # Report model summary
    count_parameters(w3dip)

    # Losses
    mse_ = torch.nn.MSELoss().to(device)
    ssim_ = SSIM3D().to(device)

    # Define Optimizer
    optimizer_ = torch.optim.Adam(
        [{'params': w3dip.image_gen.parameters()}, {'params': w3dip.kernel_gen.parameters(), 'lr': 1e-4}], lr=LR
    )
    scheduler_ = MultiStepLR(optimizer_, milestones=[2000, 3000, 4000], gamma=0.5)  # learning rates
    loss_history_ = {
        'total': [],
        'data_fitting_term': [],
        'wiener_term': [],
        'gen_kernel_similarity_to_init_kernel': [],
        'l2_reg_kernel': []
    }

    return w3dip, mse_, ssim_, optimizer_, scheduler_, loss_history_


if __name__ == '__main__':
    args, config_args = parse_arguments_and_load_config_file()
    input_cfg = config_args['input']
    img_generator_cfg = config_args['model_definition']['image_generator']
    kernel_generator_cfg = config_args['model_definition']['kernel_generator']
    train_cfg = config_args['training']
    outputs_cfg = config_args['output']

    # General params
    LR = train_cfg['lr']
    num_iter = train_cfg['steps']
    mse_to_ssim_step = train_cfg['mse_to_ssim_step']
    wk = train_cfg['loss_fn']['wk']

    kernel_size_estimate = tuple(kernel_generator_cfg['kernel_estimated_size'])
    interCNN_feature_maps = tuple(img_generator_cfg['feature_maps'])
    print(f"Unet structure: {interCNN_feature_maps}")

    save_frequency_schedule = tuple(outputs_cfg['checkpoint_frequencies'])

    base_output_dir = os.path.join(*outputs_cfg['dir'])
    input_vol_dir = os.path.join(*input_cfg['input_vols_dir'])
    kernel_dir = os.path.join(*input_cfg['kernels_dir'])

    # Load volume to fit
    dataset = SimulatedBlurDataset(
        input_volume_dir=input_vol_dir,
        kernels_dir=kernel_dir,
        device=device,
        dtype=np.float32
    )

    for (ground_truth_volume_name, ground_truth_volume), blurred_volumes in dataset:
        target_patch_num_channels = ground_truth_volume.size()[0]

        # Define model
        w3dip, mse, ssim, optimizer, scheduler, loss_history = define_model_objects(
            target_patch_num_channels=ground_truth_volume.size()[0],
            target_patch_spatial_size=tuple(ground_truth_volume.size()[1:]),
            interCNN_feature_maps=interCNN_feature_maps,
            kernel_gen_noise_input_size=kernel_generator_cfg['net_noise_input_size'],
            kernel_gen_num_hidden=kernel_generator_cfg['num_hidden_units'],
            estimated_kernel_shape=kernel_size_estimate
        )

        ground_truth_output_dir = os.path.join(
            base_output_dir,
            ground_truth_volume_name,
            f'{len(interCNN_feature_maps)}L_' + '_'.join([str(feat_map) for feat_map in interCNN_feature_maps]),
            f'wk_{wk}'
        )

        # Save the ground truth volume
        os.makedirs(ground_truth_output_dir, exist_ok=True)
        store_volume_nii_gz(
            vol_array=ground_truth_volume.cpu().detach().numpy(),
            volume_filename=f"ground_truth_volume",
            output_dir=ground_truth_output_dir)

        # Iterate over the blurring kernels to try to deconvolve
        for kernel_name, kernel, blurred_volume in blurred_volumes:
            # Create dir where the results will be stored
            output_dir = os.path.join(
                ground_truth_output_dir,
                kernel_name
            )
            os.makedirs(output_dir, exist_ok=True)

            # Save the blurred volume
            store_volume_nii_gz(
                vol_array=ground_truth_volume.cpu().detach().numpy(),
                volume_filename=f"blured_volume_w_{kernel_name}",
                output_dir=output_dir)

            # Initialization for outer-loop
            save_frequency_schedule_loop = list(save_frequency_schedule)
            save_freq_change, save_freq = save_frequency_schedule_loop.pop(0)

            for step in tqdm(range(num_iter)):
                # Forward pass
                out_x, out_k, out_y = w3dip()

                # Measure loss
                l2_reg = wk * l2_regularization(out_k)

                data_fitting_term = mse(out_y.squeeze_(), blurred_volume.squeeze_()) if step < mse_to_ssim_step else \
                    1 - ssim(out_y, blurred_volume.reshape([1, 1] + list(blurred_volume.shape)))

                loss = l2_reg + data_fitting_term
                report_memory_usage(things_in_gpu="Model and Maps")

                # Backprop
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Store loss values
                loss_history['total'].append(loss.item())
                loss_history['data_fitting_term'].append(data_fitting_term.item())
                loss_history['wiener_term'].append(0)
                loss_history['gen_kernel_similarity_to_init_kernel'].append(0)
                loss_history['l2_reg_kernel'].append(l2_reg.item())

                # Save intermediate outputs
                if step > save_freq_change and len(save_frequency_schedule_loop) > 0:
                    save_freq_change, save_freq = save_frequency_schedule_loop.pop(0)

                if step % save_freq == 0:
                    checkpoint_outputs(
                        blurred_vol_estimate=out_y, sharpened_vol_estimate=out_x.squeeze_(),
                        kernel_estimate=out_k.squeeze_(), output_dir=output_dir,
                        patch_filename=f'step_{step}'
                    )

                    # Store loss history
                    pd.DataFrame(loss_history).to_csv(os.path.join(output_dir, 'loss_history.csv'))

                # Log what happens at the last step
                checkpoint_outputs(
                    blurred_vol_estimate=out_y, sharpened_vol_estimate=out_x.squeeze_(),
                    kernel_estimate=out_k.squeeze_(),
                    output_dir=output_dir, patch_filename=f'step_{step}'
                )

                # Store loss history
                pd.DataFrame(loss_history).to_csv(os.path.join(output_dir, 'loss_history.csv'))

                # Clean up
                del out_x
                del out_y
                del out_k
                del w3dip
                torch.cuda.empty_cache()
