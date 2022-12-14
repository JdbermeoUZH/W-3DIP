from typing import Union, Tuple

import torch
from torch import nn

from model.InputNoise import InputNoise
from model.KernelGenerator import KernelGenerator
from model.ImageGenerator import ImageGenerator, ImageGeneratorInterCNN3D


def l2_regularization(tensor):
    return torch.sum(tensor ** 2)


class W3DIP(nn.Module):
    def __init__(
            self,
            image_gen: ImageGenerator,
            kernel_gen: KernelGenerator,
    ):
        super(W3DIP, self).__init__()
        self.image_gen = image_gen
        self.kernel_gen = kernel_gen

    def forward(self):
        sharp_img_estimate = self.image_gen()
        blur_kernel_estimate = self.kernel_gen().view(-1, 1, * self.kernel_gen.get_estimated_kernel_size())
        blurr_img_estimate = nn.functional.conv3d(sharp_img_estimate, blur_kernel_estimate, padding='same', bias=None)

        return sharp_img_estimate, blur_kernel_estimate, blurr_img_estimate

    def to_device(self, device: torch.device):
        self.image_gen.input_noise.to_device(device)
        self.kernel_gen.input_noise.to_device(device)
        self.to(device)


class W3DIPMultiPatches(nn.Module):
    def __init__(
            self,
            target_patch_spatial_size: Tuple[int, ...],
            num_output_channels: int,
            num_feature_maps_unet: Tuple[int, ...],
            num_patches_to_fit: int,
            kernel_gen: KernelGenerator
    ):
        super(W3DIPMultiPatches).__init__()
        self.kernel_gen = kernel_gen
        self.image_generators = []

        for patch_i in range(num_patches_to_fit):
            self.image_generators.append(
                ImageGeneratorInterCNN3D(
                    num_output_channels=num_output_channels,
                    output_spatial_size=target_patch_spatial_size,
                    input_noise=InputNoise(spatial_size=target_patch_spatial_size, num_channels=8, method='noise'),
                    downsampling_output_channels=num_feature_maps_unet
                )
            )

    def forward(self):
        sharp_img_estimates = (image_generator() for image_generator in self.image_generators)
        blur_kernel_estimate = self.kernel_gen().view(-1, 1, * self.kernel_gen.get_estimated_kernel_size())
        blurr_img_estimates = (
            nn.functional.conv3d(sharp_img_estimate, blur_kernel_estimate, padding='same', bias=None)
            for sharp_img_estimate in sharp_img_estimates
        )

        return sharp_img_estimates, blur_kernel_estimate, blurr_img_estimates

    def to_device(self, device: torch.device):
        for image_generator in self.image_generators:
            image_generator.input_noise.to_device(device)
        self.kernel_gen.input_noise.to_device(device)
        self.to(device)


