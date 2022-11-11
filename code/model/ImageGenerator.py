from typing import Optional, Union, Tuple

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
    def __init__(self, in_channels, out_channels, dropout_rate: Optional[float]=None):
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
            num_output_channels,
            input_noise_spatial_size: Union[int, Tuple[int, ...]],
            input_noise_num_channels: int = 8,
            input_noise_reg_noise_std: float = 0.001,
            upsample_strategy: str = 'bilinear',
            **kwargs,
    ):
        super(ImageGenerator, self).__init__()
        self.input = InputNoise(
            spatial_size=input_noise_spatial_size, num_channels=input_noise_num_channels,
            reg_noise_std=input_noise_reg_noise_std, **kwargs
        )

        # Encoding part
        self.enc1 = _EncoderBlock3D(input_noise_num_channels, 64)
        self.enc2 = _EncoderBlock3D(64, 128)
        #self.enc3 = _EncoderBlock(128, 256)
        #self.enc4 = _EncoderBlock(256, 512, dropout=True)

        # Center
        #self.center = _DecoderBlock(512, 1024, 512)
        self.center = _DecoderBlock3D(128, 256, 128, upsample_strategy=upsample_strategy)

        # Decoding blocks
        #self.dec4 = _DecoderBlock(1024, 512, 256)
        #self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock3D(256, 128, 64, upsample_strategy=upsample_strategy)
        self.dec1 = _DecoderBlock3D(128, 64, None, upsample_strategy=None)
        self.final = nn.Conv3d(64, num_output_channels, kernel_size=1)
        initialize_weights(self)

    def forward(self):
        x = self.input()
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        #enc3 = self.enc3(enc2)
        #enc4 = self.enc4(enc3)
        #center = self.center(enc4)
        center = self.center(enc2)
        #print(f'x.shape: {x.shape}')
        #print(f'enc1.shape: {enc1.shape}')
        #print(f'enc2.shape: {enc2.shape}')
        #print(f'center.shape: {center.shape}')
        #dec4 = self.dec4(torch.cat([center, F.interpolate(enc4, center.size()[2:], mode='bilinear', align_corners=True)], 1))
        #dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, dec4.size()[2:], mode='bilinear', align_corners=True)], 1))
        dec2 = self.dec2(torch.cat([center, F.interpolate(enc2, center.size()[2:], mode='trilinear', align_corners=True)], 1))
        #dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear', align_corners=True)], 1))
        dec1 = self.dec1(
            torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='trilinear', align_corners=True)], 1))
        final = self.final(dec1)

        return F.interpolate(final, x.size()[2:], mode='trilinear', align_corners=True)


if __name__ == '__main__':
    # Only for debugging and understanding the mappings
    example_target_shape = (3, 64, 64, 128)
    net = ImageGenerator(
        num_output_channels=example_target_shape[0],
        input_noise_spatial_size=example_target_shape[1:],
        upsample_strategy='transposed_conv'
    )

    ex_output = torch.squeeze(net())

    assert tuple(ex_output.shape) == example_target_shape
