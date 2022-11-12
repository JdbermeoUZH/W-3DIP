from typing import Tuple, Union, List

import numpy as np
from torch import nn

from model.InputNoise import InputNoise


class KernelGenerator(nn.Module):
    def __init__(
            self,
            noise_input_size: int = 200,
            num_hidden: int = 1000,
            estimated_kernel_shape: Tuple[int, ...] = (5, 5, 10),
            **kwargs,
    ):
        super(KernelGenerator, self).__init__()
        self.input_noise = InputNoise(
            spatial_size=(1, 1), num_channels=noise_input_size,
            reg_noise_std=0, **kwargs
        )

        self.model = nn.Sequential(*[
            nn.Linear(noise_input_size, num_hidden, bias=True),
            nn.ReLU6(),
            nn.Linear(num_hidden, np.prod(estimated_kernel_shape)),
            nn.Softmax()
        ])

        self.estimated_kernel_size = estimated_kernel_shape

    def forward(self):
        x = self.input_noise().squeeze_()
        return self.model(x)

    def get_estimated_kernel_size(self):
        return self.estimated_kernel_size
