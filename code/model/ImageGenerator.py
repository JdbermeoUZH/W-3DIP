from collections import OrderedDict
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from model.InputNoise import InputNoise


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class _ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dropout_rate: Optional[float] = None):
        super(_ConvBlock3D, self).__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            # TODO: Add NolocalBlock stuff from Ren et. al
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if dropout_rate:
            layers.append(nn.Dropout(dropout_rate))

        self.convBlock3D = nn.Sequential(*layers)

    def forward(self, x):
        return self.convBlock3D(x)


class _EncoderBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate: Optional[float] = None):
        super(_EncoderBlock3D, self).__init__()
        layers = [
            _ConvBlock3D(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dropout_rate=dropout_rate),
            nn.MaxPool3d(kernel_size=2, stride=2)
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock3D(nn.Module):
    def __init__(self, in_channels: int, middle_channels: int, out_channels: Optional[int],
                 kernel_size: int = 3, upsample_strategy: Optional[str] = 'trilinear'):
        super(_DecoderBlock3D, self).__init__()
        layers = [
            _ConvBlock3D(in_channels=in_channels, out_channels=middle_channels, kernel_size=kernel_size),
        ]

        if upsample_strategy == 'transposed_conv':
            layers.append(
                nn.ConvTranspose3d(middle_channels, out_channels, kernel_size=2, stride=2),
            )
        elif upsample_strategy in ['nearest', 'trilinear', 'bicubic']:
            layers.append(
                nn.Upsample(scale_factor=2, mode=upsample_strategy),
            )

        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class ImageGenerator(nn.Module):
    def __init__(
            self,
            num_output_channels: int,
            output_spatial_size: Tuple[int, ...],
            input_noise: InputNoise
    ):
        super(ImageGenerator, self).__init__()
        self.input_noise = input_noise
        self.num_output_channels = num_output_channels
        self.output_spatial_size = output_spatial_size

    def forward(self):
        x = self.input()
        return x

    def test_forward_pass_dimensions(self):
        assert self.forward().shape[-3:] == self.output_spatial_size
        assert self.forward().shape[-4] == self.num_output_channels


class ImageGeneratorInterCNN3D(ImageGenerator):
    def __init__(
            self,
            num_output_channels: int,
            output_spatial_size: Tuple[int, ...],
            input_noise: InputNoise,
            downsampling_output_channels: Tuple[int, ...],
            center_layer_scale_num_channels: int = 2
    ):
        super(ImageGeneratorInterCNN3D, self).__init__(
            num_output_channels=num_output_channels, output_spatial_size=output_spatial_size, input_noise=input_noise)

        # In this model the output has the same spatial dimensions as the input
        assert self.input_noise.get_spatial_size() == output_spatial_size
        self.num_enc_layers = len(downsampling_output_channels)

        layers = OrderedDict()

        # Encoding part
        prev_enc_layer_channels = self.input_noise.get_num_channels()
        for i, out_channels in enumerate(downsampling_output_channels):
            layers[f'enc{i + 1}'] = _EncoderBlock3D(prev_enc_layer_channels, out_channels)
            prev_enc_layer_channels = out_channels

        last_enc_out_channels = prev_enc_layer_channels

        # Center
        center_middle_channels = last_enc_out_channels * center_layer_scale_num_channels
        center_out_channels = last_enc_out_channels
        layers['center'] = _DecoderBlock3D(
            last_enc_out_channels, center_middle_channels, center_out_channels,
            upsample_strategy='transposed_conv'
        )

        # Decoding part
        prev_out_enc_channels = None
        for i, out_enc_channels in enumerate(downsampling_output_channels):
            if i + 1 == 1:
                # Last decoding layer does not upsample
                layers[f'dec{i + 1}'] = _DecoderBlock3D(
                    2 * out_enc_channels, out_enc_channels, prev_out_enc_channels, upsample_strategy=None)
            else:
                # Decoding layers that upsample
                layers[f'dec{i + 1}'] = _DecoderBlock3D(
                    2 * out_enc_channels, out_enc_channels, prev_out_enc_channels, upsample_strategy='transposed_conv')

            prev_out_enc_channels = out_enc_channels

        # Output head with 1x1 convolutions
        last_dec_layer_out_channels = downsampling_output_channels[0]
        layers['output'] = nn.Conv3d(last_dec_layer_out_channels, num_output_channels, kernel_size=1)

        for layer_name, layer in layers.items():
            self.add_module(layer_name, layer)

    def forward(self):
        mappings = {}

        # Map encoding layers
        current_input = self.input_noise()
        # print(f'current_input: {current_input.shape}')
        for level_i in range(1, self.num_enc_layers + 1):
            enc_layer_i = self.get_submodule(f'enc{level_i}')
            mappings[f'enc{level_i}'] = enc_layer_i(current_input)
            current_input = mappings[f'enc{level_i}']
            # print(f'enc{level_i}: {current_input.shape}')

        # Map center layer
        mappings[f'center'] = self.center(current_input)
        # print(f'center: {mappings[f"center"].shape}')

        # Map decoding layers
        prev_dec_layer = mappings[f'center']

        for level_i in reversed(list(range(1, self.num_enc_layers + 1))):
            dec_layer_i = self.get_submodule(f'dec{level_i}')
            prev_enc_layer = mappings[f'enc{level_i}']
            mappings[f'dec{level_i}'] = dec_layer_i(torch.cat([
                prev_dec_layer,
                F.interpolate(prev_enc_layer, prev_dec_layer.size()[2:], mode='trilinear', align_corners=True)
            ], 1)
            )
            prev_dec_layer = mappings[f'dec{level_i}']
            # print(f'dec{level_i}: {prev_dec_layer.shape}')

        final = self.output(prev_dec_layer)
        # print(f'final: {final.shape}')

        return F.interpolate(final, self.input_noise.get_spatial_size(), mode='trilinear', align_corners=True)


if __name__ == '__main__':
    # Only for debugging and understanding the mappings
    example_target_shape = (3, 32, 32, 256)

    intercnn3d = ImageGeneratorInterCNN3D(
        num_output_channels=example_target_shape[0],
        output_spatial_size=example_target_shape[1:],
        input_noise=InputNoise(spatial_size=example_target_shape[1: ], num_channels=8, method='noise'),
        downsampling_output_channels=(64,)
    )

    print(intercnn3d().shape)
    intercnn3d.test_forward_pass_dimensions()
    assert intercnn3d().shape[2:] == example_target_shape[1:]
