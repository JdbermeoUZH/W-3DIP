import os
import yaml
import argparse
from typing import Tuple

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset.SimulatedBlurDataset import SimulatedBlurDataset
from model.InputNoise import InputNoise
from model.W3DIP import W3DIP
from model.ImageGenerator import ImageGeneratorInterCNN3D
from model.KernelGenerator import KernelGenerator
from train.W3DIPTrainer import W3DIPTrainer
from utils.common_utils import store_volume_nii_gz

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


def plot_figures_store_metrics(metrics_):
    # Wrange the metrics logged into a dataframe
    dataframes = []
    baseline_df_list = []
    for vol_name, metrics_of_different_kernels in metrics_.items():
        for kernel_type, metrics_per_kernel in metrics_of_different_kernels.items():
            sharp_image_metrics_df = pd.DataFrame(metrics_per_kernel['sharp_volume_estimation']) \
                .set_index('step')
            kernel_metrics = pd.DataFrame(metrics_per_kernel['kernel_estimation']) \
                .set_index('step')
            sharp_image_metrics_df = sharp_image_metrics_df.join(kernel_metrics.mse.rename('kernel_mse'))
            sharp_image_metrics_df['vol_name'] = vol_name
            sharp_image_metrics_df['kernel_type'] = kernel_type
            sharp_image_metrics_df = sharp_image_metrics_df.reset_index().set_index(['vol_name', 'kernel_type', 'step'])
            dataframes.append(sharp_image_metrics_df)

            baseline_df_list.append({'vol_name': vol_name, 'kernel_type': kernel_type} | metrics_per_kernel['baseline'])

    # Store the model metrics and baseline performance metrics
    results_df = pd.concat(dataframes)
    results_df.to_csv(os.path.join(base_output_dir, 'consolidated_performance_accross_experiments.csv'))

    baselines_df = pd.DataFrame(baseline_df_list).set_index('vol_name').sort_index()
    baselines_df.to_csv(os.path.join(base_output_dir, 'baseline_performance_across_experiments.csv'))
    kernel_names = [kernel_name_ for kernel_name_ in results_df.index.get_level_values(1).drop_duplicates()]

    max_steps = results_df.index.get_level_values(2).max()

    # Generate plots
    for kernel in kernel_names:
        for metric in results_df.columns:
            results_df.sort_index().loc[(slice(None), slice(kernel)), :].reset_index() \
                .pivot_table(index='step', columns='vol_name', values=[metric]) \
                .plot(figsize=(10, 5), logy=True, ylabel=metric,
                      title=f"Performance for the {kernel} kernel: Estimated Sharp Volume")

            colors = [line.get_color() for line in plt.gca().lines]

            plt.vlines(x=mse_to_ssim_step, ymin=0,
                       ymax=results_df.sort_index().loc[(slice(None), slice(kernel)), metric].max(),
                       linestyle='--')

            if metric != 'kernel_mse':
                print(metric)
                for i, (img_name, value) in enumerate(baselines_df[baselines_df.kernel_type == kernel][metric].items()):
                    print(img_name)
                    plt.hlines(y=value, xmin=0, xmax=max_steps, linestyle='--', alpha=0.55, color=colors[i])

            plt.savefig(os.path.join(base_output_dir, f'performance_{metric}_w_kernel_{kernel}.jpg'))


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
    dataset = SimulatedBlurDataset(
        input_volume_dir=input_vol_dir,
        kernels_dir=kernel_dir,
        device=device,
        dtype=np.float32
    )

    metrics = {}
    for (ground_truth_volume_name, ground_truth_volume), blurred_volumes in dataset:
        report_image_name_str = f"Running W3DIP for image: {ground_truth_volume_name}"
        print(f"{report_image_name_str}\n{len(report_image_name_str) * '#'}")

        target_patch_num_channels = ground_truth_volume.size()[0]

        ground_truth_output_dir = os.path.join(
            base_output_dir,
            ground_truth_volume_name,
            f'{len(interCNN_feature_maps)}L_' + '_'.join([str(feat_map) for feat_map in interCNN_feature_maps]),
            f'wk_{wk}'
        )

        # Save the ground truth volume
        os.makedirs(ground_truth_output_dir, exist_ok=True)
        store_volume_nii_gz(
            vol_array=ground_truth_volume[0].cpu().detach().numpy(),
            volume_filename=f"ground_truth_volume.nii.gz",
            output_dir=ground_truth_output_dir)

        target_patch_spatial_size = tuple(ground_truth_volume.size()[1:])

        metrics[ground_truth_volume_name] = {}

        # Iterate over the blurring kernels to try to deconvolve
        for kernel_name, kernel, blurred_volume in blurred_volumes:
            report_kernel_type = f"Running for kernel: {kernel_name}"
            print(f"{report_kernel_type}\n{len(report_kernel_type) * '-'}\n")

            # Create dir where the results will be stored
            output_dir = os.path.join(
                ground_truth_output_dir,
                kernel_name
            )
            os.makedirs(output_dir, exist_ok=True)

            # Save the blurred volume
            store_volume_nii_gz(
                vol_array=blurred_volume[0].cpu().detach().numpy(),
                volume_filename=f"blured_volume_w_{kernel_name}.nii.gz",
                output_dir=output_dir)

            # Define model
            w3dip = W3DIP(
                image_gen=ImageGeneratorInterCNN3D(
                    num_output_channels=ground_truth_volume.size()[0],
                    output_spatial_size=target_patch_spatial_size,
                    input_noise=InputNoise(spatial_size=target_patch_spatial_size, num_channels=8, method='noise'),
                    downsampling_output_channels=interCNN_feature_maps
                ),

                kernel_gen=KernelGenerator(
                    noise_input_size=kernel_generator_cfg['net_noise_input_size'],
                    num_hidden=kernel_generator_cfg['num_hidden_units'],
                    estimated_kernel_shape=kernel_size_estimate
                )
            )

            trainer = W3DIPTrainer(
                w3dip=w3dip,
                device=device,
                lr_img_network=LR,
                lr_kernel_network=1e-4,
                lr_schedule_params={"milestones": [2000, 3000, 4000], "gamma": 0.5},
                w_k=wk
            )

            trainer.fit_no_guidance(
                blurred_volume=blurred_volume,
                sharp_volume_ground_truth=ground_truth_volume,
                kernel_ground_truth=kernel,
                num_steps=num_iter,
                mse_to_ssim_step=mse_to_ssim_step,
                checkpoint_schedule=save_frequency_schedule,
                checkpoint_base_dir=output_dir
            )

            metrics[ground_truth_volume_name][kernel_name] = {
                'sharp_volume_estimation': trainer.image_estimate_metrics,
                'kernel_estimation': trainer.kernel_estimate_metrics,
                'baseline': trainer.baseline_blurr_to_gt_metrics
            }

    plot_figures_store_metrics(metrics)
