import json
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
from model.W3DIP import W3DIP, W3DIPMultiPatch
from model.KernelGenerator import KernelGenerator
from train.W3DIPMultiPatchTrainer import W3DIPMultiPatchTrainer
from utils.common_utils import store_volume_nii_gz, set_seed

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


def plot_figures_store_metrics_per_experiment(metrics_per_exp, baseline_metrics, output_dir_, kernel_name,
                                              mse_to_ssim_step_):
    # Wrangle the metrics logged into a dataframe
    sharp_image_metrics_df = pd.DataFrame(metrics_per_exp['sharp_volume_estimation'])\
        .set_index(['image_name', 'step']).sort_index()

    sharp_image_metrics_df.to_csv(os.path.join(output_dir_, 'consolidated_performance_accross_experiments.csv'))

    kernel_metrics_df = pd.DataFrame(metrics_per_exp['kernel_estimation']).set_index('step').sort_index()

    baseline_df = pd.DataFrame(baseline_metrics).set_index('image_name').sort_index()

    plot_volume_performance_per_metric(baseline_df, sharp_image_metrics_df, kernel_name, output_dir_)

    # Generate plots for the kernel estimates
    plot_kernel_estimate_performance(
        kernel_metrics_df=kernel_metrics_df, kernel_name=kernel_name, mse_to_ssim_step_=mse_to_ssim_step_,
        output_dir_=output_dir_, metric_name='mse')


def plot_volume_performance_per_metric(
        baseline_df: pd.DataFrame,
        sharp_vol_metrics_df: pd.DataFrame,
        kenel_name: str, output_dir_: str,
        with_conf_interval: bool = False):

    max_steps = sharp_vol_metrics_df.index.get_level_values(1).max()
    list_of_metrics = list(set([col.split('_')[0] for col in sharp_vol_metrics_df.columns]))

    # Generate plots for the sharp image estimates
    for metric_name in list_of_metrics:
        y_col = f'{metric_name}_mean' if with_conf_interval else metric_name
        y_col_max = f'{metric_name}_mean_high' if with_conf_interval else metric_name
        fig_title = f"Sharp volume {metric_name}: Performance for the kernel {kenel_name}"

        sharp_vol_metrics_df.reset_index() \
            .pivot_table(index='step', columns='image_name', values=[y_col]) \
            .plot(figsize=(10, 5), logy=True, ylabel=y_col,
                  title=fig_title)

        colors = [line.get_color() for line in plt.gca().lines]

        # Plot change to ssim loss instead of mse
        plt.vlines(x=mse_to_ssim_step, ymin=0,
                   ymax=sharp_vol_metrics_df.loc[:, y_col_max].max(),
                   linestyle='--')

        for i, (img_name, value) in enumerate(baseline_df[metric_name].items()):
            # Plot baselines
            plt.hlines(y=value, xmin=0, xmax=max_steps, linestyle='--', alpha=0.55, color=colors[i])

            if with_conf_interval:
                # Plot confidence intervals
                sharp_vol_i_metrics_df = sharp_vol_metrics_df.loc[img_name]
                plt.fill_between(
                    sharp_vol_i_metrics_df.index,
                    sharp_vol_i_metrics_df[f'{metric_name}_mean_low'],
                    sharp_vol_i_metrics_df[f'{metric_name}_mean_high'],
                    color=colors[i], alpha=.1
                )

        plt.savefig(
            os.path.join(output_dir_, f'{fig_title}.jpg'))
        plt.close()


def plot_kernel_estimate_performance(kernel_metrics_df: pd.DataFrame, kernel_name: str, mse_to_ssim_step_: int,
                                     output_dir_: str, metric_name: str, with_conf_interval: bool = False):
    figure_name = f"Kernel {kernel_name} estimate: {metric_name} "
    y_col = f'{metric_name}_mean' if with_conf_interval else metric_name
    y_col_max = f'{metric_name}_mean_high' if with_conf_interval else metric_name

    kernel_metrics_df[y_col].plot(figsize=(10, 5), logy=True, ylabel=metric_name, title=figure_name)
    plt.vlines(x=mse_to_ssim_step_, ymin=0, ymax=kernel_metrics_df[y_col_max].max(), linestyle='--')

    if with_conf_interval:
        plt.fill_between(
            kernel_metrics_df.index,
            kernel_metrics_df[f'{metric_name}_mean_low'],
            kernel_metrics_df[f'{metric_name}_mean_high'],
            color='b', alpha=.1
        )
    plt.savefig(os.path.join(output_dir_, f'{figure_name}.jpg'))
    plt.close()


def concatenate_experiment_metrics(kernel_metrics_dictionary):

    sharp_volume_estimation_metric_dfs = []
    kernel_estimation_metric_dfs = []

    for random_state_exp, metrics_dict in kernel_metrics_dictionary.items():
        sharp_vol_metrics_df_i = pd.DataFrame(metrics_dict['sharp_volume_estimation'])
        sharp_vol_metrics_df_i['exp'] = random_state_exp
        sharp_vol_metrics_df_i = sharp_vol_metrics_df_i.set_index(['exp', 'image_name', 'step']).sort_index()

        sharp_volume_estimation_metric_dfs.append(sharp_vol_metrics_df_i)

        kernel_metrics_df_i = pd.DataFrame(metrics_dict['kernel_estimation'])
        kernel_metrics_df_i['exp'] = random_state_exp
        kernel_metrics_df_i = kernel_metrics_df_i.set_index(['exp', 'step']).sort_index()
        kernel_estimation_metric_dfs.append(kernel_metrics_df_i)

    sharp_volume_estimation_metric_df = pd.concat(sharp_volume_estimation_metric_dfs)
    kernel_estimation_metric_df = pd.concat(kernel_estimation_metric_dfs)

    # Calculate mean and standard deviation across experiments
    sharp_volume_estimation_metric_df_per_volume = sharp_volume_estimation_metric_df.groupby(level=[1, 2]).agg(
        [np.mean, np.std, lambda x: np.mean(x) - np.std(x), lambda x: np.mean(x) + np.std(x)]).rename(
        columns={'<lambda_0>': 'mean_low', '<lambda_1>': 'mean_high'})
    sharp_volume_estimation_metric_df_per_volume.columns = sharp_volume_estimation_metric_df_per_volume.columns \
        .map('_'.join).str.strip('_')

    sharp_volume_estimation_metric_df_agg = sharp_volume_estimation_metric_df.groupby(level=[2]).agg(
        [np.mean, np.std, lambda x: np.mean(x) - np.std(x), lambda x: np.mean(x) + np.std(x)]).rename(
        columns={'<lambda_0>': 'mean_low', '<lambda_1>': 'mean_high'})
    sharp_volume_estimation_metric_df_agg.columns = sharp_volume_estimation_metric_df_agg.columns \
        .map('_'.join).str.strip('_')

    kernel_estimation_metric_df = kernel_estimation_metric_df.groupby(level=[1]).agg(
        [np.mean, np.std, lambda x: np.mean(x) - np.std(x), lambda x: np.mean(x) + np.std(x)]).rename(
        columns={'<lambda_0>': 'mean_low', '<lambda_1>': 'mean_high'})
    kernel_estimation_metric_df.columns = kernel_estimation_metric_df.columns.map('_'.join) \
        .str.strip('_')

    return kernel_estimation_metric_df, sharp_volume_estimation_metric_df_per_volume,\
        sharp_volume_estimation_metric_df_agg


def plot_aggregated_volume_metrics(agg_baseline_df, sharp_volume_est_agg_df, output_dir_):

    for metric_name in agg_baseline_df.columns:
        fig_title = f'Sharp volume {metric_name}: Performance aggregated across volumes'
        steps = sharp_volume_est_agg_df.index
        baseline_values = np.array(
            [agg_baseline_df.loc['mean', metric_name]] * sharp_volume_est_agg_df.shape[0])
        baseline_std = agg_baseline_df.loc['std', metric_name]
        fig, ax = plt.subplots()

        # Plot baseline
        ax.plot(steps, baseline_values, label='averaged baseline')
        ax.fill_between(steps, (baseline_values - baseline_std), (baseline_values + baseline_std), color='b', alpha=.1)

        # Plot estimate
        estim_perf = sharp_volume_est_agg_df[f'{metric_name}_mean']
        estim_perf_high = sharp_volume_est_agg_df[f'{metric_name}_mean_high']
        estim_perf_low = sharp_volume_est_agg_df[f'{metric_name}_mean_low']

        ax.plot(steps, estim_perf, label='W-3DIP averaged estimate')
        ax.fill_between(steps, estim_perf_low, estim_perf_high, alpha=.1)

        ax.set_title(fig_title)
        ax.set_yscale('log')
        ax.set_ylabel(metric_name)
        ax.set_xlabel('steps')
        plt.legend(loc="upper left")
        plt.savefig(os.path.join(output_dir_, f'{fig_title}.jpg'))


def record_agg_metrics_to_report(baseline_agg_df_, kernel_est_metric_df_):
    # For sharp estimate of volumes
    for metric_name in baseline_agg_df_.columns:
        if metric_name in ['mse']:
            best_score_idx = sharp_volume_est_metric_agg_df[f'{metric_name}_mean'].idxmin()
        else:
            best_score_idx = sharp_volume_est_metric_agg_df[f'{metric_name}_mean'].idxmax()

        best_score = f'{sharp_volume_est_metric_agg_df[f"{metric_name}_mean"].loc[best_score_idx]: 0.4f} $\pm$ ' \
                     f'{sharp_volume_est_metric_agg_df[f"{metric_name}_std"].loc[best_score_idx]: 0.4f}'
        last_score = f'{sharp_volume_est_metric_agg_df.iloc[-1][f"{metric_name}_mean"]: 0.4f} $\pm$ ' \
                     f'{sharp_volume_est_metric_agg_df.iloc[-1][f"{metric_name}_std"]: 0.4f}'
        baseline_socre = f'{baseline_agg_df_.loc["mean", metric_name]: 0.4f} $\pm$ ' \
                         f'{baseline_agg_df_.loc["std", metric_name]: 0.4f}'

        metrics_to_report[metric_name][blurring_kernel_name] = {
            'best_score': best_score, 'last_score': last_score, 'baseline_score': baseline_socre}
    # For mse of kernel
    best_kernel_mse_idx = kernel_est_metric_df_.mse_mean.idxmin()
    best_kernel_mse = f'{kernel_est_metric_df_.loc[best_kernel_mse_idx, "mse_mean"]: 0.4f} $\pm$ ' \
                      f'{kernel_est_metric_df_.loc[best_kernel_mse_idx, "mse_std"]: 0.4f}'
    last_kernel_mse = f'{kernel_est_metric_df_.iloc[-1]["mse_mean"]: 0.4f} $\pm$ ' \
                      f'{kernel_est_metric_df_.iloc[-1]["mse_std"]: 0.4f}'
    metrics_to_report['kernel_mse'][blurring_kernel_name] = {
        'best_score': best_kernel_mse, 'last_score': last_kernel_mse}


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
    metrics_to_report = {}
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
            random_state = random.randint(0, 1000)
            set_seed(random_state)

            checkpoint_volumes = repetition_i == 0
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

            # Metric reporting
            if repetition_i == 0:
                metrics[blurring_kernel_name]['baseline'] = trainer.baseline_blurr_to_gt_metrics

            # Initialize keys of metrics to report
            if len(metrics_to_report) == 0:
                for metric_name in list(trainer.baseline_blurr_to_gt_metrics.keys() - {'image_name'}):
                    metrics_to_report[metric_name] = {}

                metrics_to_report['kernel_mse'] = {}

            metrics[blurring_kernel_name][f'seed_{random_state}'] = {
                    'sharp_volume_estimation': trainer.image_estimate_metrics,
                    'kernel_estimation': trainer.kernel_estimate_metrics,
            }

            plot_figures_store_metrics_per_experiment(
                metrics[blurring_kernel_name][f'seed_{random_state}'],
                metrics[blurring_kernel_name]['baseline'],
                exp_output_dir,
                blurring_kernel_name,
                mse_to_ssim_step_=mse_to_ssim_step
            )

        # Concatenate results across experiments with different seeds
        kernel_est_metric_df, sharp_vol_est_metric_df_per_vol, sharp_volume_est_metric_agg_df =\
            concatenate_experiment_metrics(
                {k: v for k, v in metrics[blurring_kernel_name].items() if 'baseline' not in k})

        # Plot average kernel performance
        plot_kernel_estimate_performance(
            kernel_name=blurring_kernel_name, kernel_metrics_df=kernel_est_metric_df,
            mse_to_ssim_step_=mse_to_ssim_step, output_dir_=kernel_output_dir, metric_name='mse',
            with_conf_interval=True
        )

        # Plot average sharp volume performance per volume
        baseline_df = pd.DataFrame(metrics[blurring_kernel_name]['baseline']).set_index('image_name').sort_index()
        plot_volume_performance_per_metric(
            baseline_df=baseline_df,
            sharp_vol_metrics_df=sharp_vol_est_metric_df_per_vol,
            kenel_name=blurring_kernel_name,
            output_dir_=kernel_output_dir,
            with_conf_interval=True
        )

        # Plot average sharp volume performance across volumes
        baseline_agg_df = baseline_df.agg([np.mean, np.std])
        plot_aggregated_volume_metrics(baseline_agg_df, sharp_volume_est_metric_agg_df, output_dir_=kernel_output_dir)

        record_agg_metrics_to_report(baseline_agg_df, kernel_est_metric_df)

    # Store aggregated metrics as markdown tables
    for metric_name in metrics_to_report.keys():
        f = open(os.path.join(ground_truth_output_dir, f'{metric_name}.md'), "w")
        table_i = pd.DataFrame(metrics_to_report[metric_name]).to_markdown()
        f.write(table_i)
        f.close()

    # Store all metrics calculated
    json.dump(metrics, open(os.path.join(ground_truth_output_dir, 'exp_metrics.json'), 'w'), indent=4)

