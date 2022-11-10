from typing import Union, Tuple

import numpy as np
import torch
from torch import nn

from utils import common_utils


class InputNoise(nn.Module):
    def __init__(
            self,
            spatial_size: Union[int, Tuple[int, ...]],
            num_channels: int = 8,
            method: str = 'noise',
            noise_type='u',
            var=0.1,
            reg_noise_std: float = 0.001
    ):
        """Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
                    initialized in a specific way.
                    Args:
                        spatial_size: spatial size of the tensor to initialize
                        num_channels: number of channels in the tensor
                        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
                        noise_type: 'u' for uniform; 'n' for normal
                        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
                        reg_noise_std: Proportion of noise perturbation added to the input
        """
        super(InputNoise, self).__init__()

        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size, spatial_size)

        if method == 'noise':
            shape = [1, num_channels] + list(spatial_size)
            net_input = torch.zeros(shape)

            common_utils.fill_noise(net_input, noise_type)
            net_input *= var

        elif method == 'meshgrid':
            meshgrid_args = [np.arange(0, spatial_size_i) / float(spatial_size_i - 1) for spatial_size_i in
                             spatial_size]

            grid = np.meshgrid(*meshgrid_args)
            meshgrid = np.concatenate([x_i[None, :] for x_i in grid])
            net_input = common_utils.np_to_torch(meshgrid)
        else:
            assert False

        self.input = net_input
        self.reg_noise_std = reg_noise_std

    def forward(self):
        return self.input + self.reg_noise_std * torch.zeros(self.input.shape).type_as(self.input.data).normal_()

    def to_cuda(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input = self.input.to(device)
