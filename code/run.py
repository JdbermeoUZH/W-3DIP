import os
import yaml
import random
import argparse
from typing import Tuple

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset.SimulatedBlurDataset import SimulatedBlurDatasetMultiPatch
from model.InputNoise import InputNoise
from model.W3DIP import W3DIP, W3DIPMultiPatch
from model.ImageGenerator import ImageGeneratorInterCNN3D
from model.KernelGenerator import KernelGenerator
from train.W3DIPMultiPatchTrainer import W3DIPMultiPatchTrainer
from train.W3DIPTrainer import W3DIPTrainer
from utils.common_utils import store_volume_nii_gz

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# warnings.filterwarnings("ignore")


def set_seed(seed: int = 123):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


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


def plot_figures_store_metrics_per_experiment(metrics_per_exp, output_dir_, kenel_name):
    # Wrangle the metrics logged into a dataframe
    sharp_image_metrics_df = pd.DataFrame(metrics_per_exp['sharp_volume_estimation'])\
        .set_index(['image_name', 'step']).sort_index()

    sharp_image_metrics_df.to_csv(os.path.join(output_dir_, 'consolidated_performance_accross_experiments.csv'))

    kernel_metrics_df = pd.DataFrame(metrics_per_exp['kernel_estimation']).set_index('step').sort_index()

    baseline_df = pd.DataFrame(metrics_per_exp['baseline']).set_index('image_name').sort_index()

    max_steps = sharp_image_metrics_df.index.get_level_values(1).max()

    # Generate plots for the sharp image estimates
    for metric in sharp_image_metrics_df.columns:
        sharp_image_metrics_df.reset_index() \
            .pivot_table(index='step', columns='image_name', values=[metric]) \
            .plot(figsize=(10, 5), logy=True, ylabel=metric,
                  title=f"Sharp volume {metric}: Performance for the kernel {kenel_name}")

        colors = [line.get_color() for line in plt.gca().lines]

        plt.vlines(x=mse_to_ssim_step, ymin=0,
                   ymax=sharp_image_metrics_df.loc[:, metric].max(),
                   linestyle='--')

        for i, (img_name, value) in enumerate(baseline_df[metric].items()):
            plt.hlines(y=value, xmin=0, xmax=max_steps, linestyle='--', alpha=0.55, color=colors[i])

        plt.savefig(
            os.path.join(output_dir_, f'Sharp volume {metric}: Performance for the kernel {kenel_name}.jpg'))
        plt.close()

        # Generate plots for the kernel estimates
        kernel_metrics_df.mse.plot(figsize=(10, 5), logy=True, ylabel='mse',
                                   title=f"Kernel MSE: Performance for the kernel {kenel_name}")
        plt.vlines(x=mse_to_ssim_step, ymin=0, ymax=kernel_metrics_df.mse.max(), linestyle='--')
        plt.savefig(os.path.join(output_dir_, f'Kernel MSE: Performance for the kernel {kenel_name}.jpg'))
        plt.close()


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

    # Folders with volumes and kernels to test
    dataset = SimulatedBlurDatasetMultiPatch(
        input_volume_dir=input_vol_dir,
        kernels_dir=kernel_dir,
        device=device,
        dtype=np.float32
    )

    metrics = {}
    for ground_truth_volume_names, ground_truth_volumes, blurring_kernel_name, blurr_kernel, blurred_volumes in dataset:
        metrics[blurring_kernel_name] = {}

        report_image_name_str = f"Running W3DIP for kernel : {blurring_kernel_name}"
        print(f"{report_image_name_str}\n{len(report_image_name_str) * '#'}")

        print(f'Fitting the volumes: {ground_truth_volume_names}')

        target_patches_num_channels = ground_truth_volumes.size()[1]
        num_volumes_to_fit = ground_truth_volumes.size()[0]

        ground_truth_output_dir = os.path.join(
            base_output_dir,
            f'{len(interCNN_feature_maps)}L_' + '_'.join([str(feat_map) for feat_map in interCNN_feature_maps]),
            f'wk_{wk}'
        )

        # Save the ground truth volumes
        os.makedirs(ground_truth_output_dir, exist_ok=True)
        for ground_truth_volume_name, ground_truth_volume in zip(ground_truth_volume_names, ground_truth_volumes):

            store_volume_nii_gz(
                vol_array=ground_truth_volume[0].cpu().detach().numpy(),
                volume_filename=f"{ground_truth_volume_name}.nii.gz",
                output_dir=ground_truth_output_dir)

        # Create dir where the results will be stored
        kernel_output_dir = os.path.join(
            ground_truth_output_dir,
            blurring_kernel_name
        )
        blurred_volume_names = tuple(f"blurred_{ground_truth_volume_name}.nii.gz"
                                     for ground_truth_volume_name in ground_truth_volume_names)

        for repetition_i in range(train_cfg['num_repetitions']):
            random_state = random.randint(0, 100)
            set_seed(random_state)

            checkpoint_volumes = repetition_i == 0 # Only store volumes of the first iteration
            exp_output_dir = os.path.join(kernel_output_dir, f'seed_{random_state}')
            os.makedirs(exp_output_dir, exist_ok=True)

            metrics[blurring_kernel_name][f'seed_{random_state}'] = {}

            # Define model
            target_patch_spatial_size = tuple(ground_truth_volumes.size()[2:])

            w3dip = W3DIPMultiPatch(
                target_patch_spatial_size=target_patch_spatial_size,
                num_output_channels=target_patches_num_channels,
                num_feature_maps_unet=interCNN_feature_maps,
                num_patches_to_fit=num_volumes_to_fit,

                kernel_gen=KernelGenerator(
                    noise_input_size=kernel_generator_cfg['net_noise_input_size'],
                    num_hidden=kernel_generator_cfg['num_hidden_units'],
                    estimated_kernel_shape=kernel_size_estimate
                )
            )
            w3dip.to_device(device)

            trainer = W3DIPMultiPatchTrainer(
                    w3dip=w3dip,
                    device=device,
                    lr_img_network=LR,
                    lr_kernel_network=1e-4,
                    lr_schedule_params={"milestones": [2000, 3000, 4000], "gamma": 0.5},
                    w_k=wk
            )

            trainer.fit_no_guidance(
                blurred_volumes=blurred_volumes,
                blurred_volume_names=blurred_volume_names,
                sharp_volumes_ground_truth=ground_truth_volumes,
                kernel_ground_truth=blurr_kernel,
                kernel_ground_truth_name=blurring_kernel_name,
                num_steps=num_iter,
                mse_to_ssim_step=mse_to_ssim_step,
                checkpoint_schedule=save_frequency_schedule,
                checkpoint_base_dir=exp_output_dir,
                checkpoint_volumes=checkpoint_volumes
            )

            metrics[blurring_kernel_name][f'seed_{random_state}'] = {
                    'sharp_volume_estimation': trainer.image_estimate_metrics,
                    'kernel_estimation': trainer.kernel_estimate_metrics,
                    'baseline': trainer.baseline_blurr_to_gt_metrics
            }

            plot_figures_store_metrics_per_experiment(
                metrics[blurring_kernel_name][f'seed_{random_state}'],
                exp_output_dir,
                blurring_kernel_name
            )
