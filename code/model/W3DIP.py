from typing import Union, Tuple

import torch
from torch import nn

from model.KernelGenerator import KernelGenerator
from model.ImageGenerator import ImageGenerator


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


