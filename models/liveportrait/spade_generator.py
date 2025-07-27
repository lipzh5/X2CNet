# coding: utf-8

"""
Spade decoder(G) defined in the paper, which input the warped feature to generate the animated image.
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm



class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 128

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out



class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, norm_G, label_nc, use_se=False, dilation=1):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.use_se = use_se
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=dilation, dilation=dilation)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        # apply spectral norm if specified
        if 'spectral' in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        # define normalization layers
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)

    def forward(self, x, seg1):
        x_s = self.shortcut(x, seg1)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg1)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg1)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg1):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg1))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)



class SPADEDecoder(nn.Module):
    def __init__(self, upscale=1, max_features=256, block_expansion=64, out_channels=64, num_down_blocks=2):
        for i in range(num_down_blocks):
            input_channels = min(max_features, block_expansion * (2 ** (i + 1)))
        self.upscale = upscale
        super().__init__()
        norm_G = 'spadespectralinstance'
        label_num_channels = input_channels  # 256

        self.fc = nn.Conv2d(input_channels, 2 * input_channels, 3, padding=1)
        self.G_middle_0 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_1 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_2 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_3 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_4 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_5 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.up_0 = SPADEResnetBlock(2 * input_channels, input_channels, norm_G, label_num_channels)
        self.up_1 = SPADEResnetBlock(input_channels, out_channels, norm_G, label_num_channels)
        self.up = nn.Upsample(scale_factor=2)

        if self.upscale is None or self.upscale <= 1:
            self.conv_img = nn.Conv2d(out_channels, 3, 3, padding=1)
        else:
            self.conv_img = nn.Sequential(
                nn.Conv2d(out_channels, 3 * (2 * 2), kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor=2)
            )

    def forward(self, feature):
        seg = feature  # Bx256x64x64
        x = self.fc(feature)  # Bx512x64x64
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x = self.G_middle_2(x, seg)
        x = self.G_middle_3(x, seg)
        x = self.G_middle_4(x, seg)
        x = self.G_middle_5(x, seg)

        x = self.up(x)  # Bx512x64x64 -> Bx512x128x128
        x = self.up_0(x, seg)  # Bx512x128x128 -> Bx256x128x128
        x = self.up(x)  # Bx256x128x128 -> Bx256x256x256
        x = self.up_1(x, seg)  # Bx256x256x256 -> Bx64x256x256

        x = self.conv_img(F.leaky_relu(x, 2e-1))  # Bx64x256x256 -> Bx3xHxW
        x = torch.sigmoid(x)  # Bx3xHxW

        return x