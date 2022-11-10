from typing import Union, Tuple

from torch import nn

from model.KernelGenerator import KernelGenerator
from model.ImageGenerator import ImageGenerator


class W3DIP(nn.Module):
    def __init__(
            self,
            img_gen_input_noise_spatial_size: Union[int, Tuple[int, ...]],
            img_gen_output_channels: int = 1,
            img_gen_input_noise_num_channels: int = 8,
            img_gen_input_noise_reg_noise_std: float = 0.001,
            img_gen_upsample_strategy: str = 'bilinear',
            kernel_net_noise_input_size: int = 200,
            kernel_net_num_hidden: int = 1000,
            estimated_kernel_shape: Tuple[int, ...] = (5, 5, 10),
    ):
        super(W3DIP, self).__init__()
        self.image_gen = ImageGenerator(
            num_output_channels=img_gen_output_channels,
            input_noise_spatial_size=img_gen_input_noise_spatial_size,
            input_noise_num_channels=img_gen_input_noise_num_channels,
            input_noise_reg_noise_std=img_gen_input_noise_reg_noise_std,
            upsample_strategy=img_gen_upsample_strategy
        )

        self.kernel_gen = KernelGenerator(
            noise_input_size=kernel_net_noise_input_size, num_hidden=kernel_net_num_hidden,
            estimated_kernel_shape=estimated_kernel_shape
        )

    def forward(self):
        return self.image_gen(), self.kernel_gen()

    def input_noises_to_cuda(self):
        self.image_gen.input.to_cuda()
        self.kernel_gen.input.to_cuda()


