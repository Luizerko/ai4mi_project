#!/usr/bin/env python3.10

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Jose Dolz

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import SqueezeExcitation

DEFAULT_NORM_GROUPS = 32
def group_count(desired_g, num_channels):
        g = min(desired_g, num_channels)
        while g > 1 and (num_channels % g) != 0:
                g -= 1
        return g

def random_weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
        elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
        elif isinstance(m, nn.GroupNorm):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

def conv_block(in_dim, out_dim, **kwconv):
        return nn.Sequential(nn.Conv2d(in_dim, out_dim, **kwconv),
                        #      nn.BatchNorm2d(out_dim),
                             nn.GroupNorm(group_count(DEFAULT_NORM_GROUPS, out_dim), out_dim),
                             nn.PReLU())


def conv_block_asym(in_dim, out_dim, *, kernel_size: int):
        return nn.Sequential(nn.Conv2d(in_dim, out_dim,
                                       kernel_size=(kernel_size, 1),
                                       padding=(2, 0)),
                             nn.Conv2d(out_dim, out_dim,
                                       kernel_size=(1, kernel_size),
                                       padding=(0, 2)),
                        #      nn.BatchNorm2d(out_dim),
                             nn.GroupNorm(group_count(DEFAULT_NORM_GROUPS, out_dim), out_dim),
                             nn.PReLU())


class C3Block(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()

        # Concentrate (local and dense): wtih aggregation via asymmetric 1x3 then 3x1 and no dilation
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        # self.norm1 = nn.BatchNorm2d(channels)
        self.norm1 = nn.GroupNorm(group_count(DEFAULT_NORM_GROUPS, channels), channels)
        self.act1  = nn.PReLU()

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
        # self.norm2 = nn.BatchNorm2d(channels)
        self.norm2 = nn.GroupNorm(group_count(DEFAULT_NORM_GROUPS, channels), channels)
        self.act2  = nn.PReLU()

        # Comprehensive (context and large receptive field): depthwise 3x3 with dilation
        self.dw = nn.Conv2d(
            channels, channels,
            kernel_size=3, padding=dilation, dilation=dilation,
            groups=channels, bias=False
        )
        self.norm3 = nn.BatchNorm2d(channels)
        self.act3  = nn.PReLU()

    def forward(self, x):
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        x = self.act3(self.norm3(self.dw(x)))
        return x


class BottleNeck(nn.Module):
        def __init__(self, in_dim, out_dim, projectionFactor,
                     *, dropoutRate=0.01, dilation=1,
                     asym: bool = False, dilate_last: bool = False,
                     attn: str = "", attn_red: int = 16, c3: bool = False):
                super().__init__()
                self.in_dim = in_dim
                self.out_dim = out_dim
                mid_dim: int = in_dim // projectionFactor

                # Main branch

                # Secondary branch
                self.block0 = conv_block(in_dim, mid_dim, kernel_size=1)

                if c3:
                        self.block1 = C3Block(mid_dim, dilation=dilation)
                else:
                        if not asym:
                                self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=dilation, dilation=dilation)
                        else:
                                self.block1 = conv_block_asym(mid_dim, mid_dim, kernel_size=5)

                self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)

                if attn == "se":
                        squeeze_channels = max(1, out_dim // attn_red)
                        self.attn = SqueezeExcitation(out_dim, squeeze_channels)
                else:
                        self.attn = nn.Identity()

                self.do = nn.Dropout(p=dropoutRate)
                self.PReLU_out = nn.PReLU()

                if in_dim > out_dim:
                        self.conv_out = conv_block(in_dim, out_dim, kernel_size=1)
                elif dilate_last:
                        self.conv_out = conv_block(in_dim, out_dim, kernel_size=3, padding=1)
                else:
                        self.conv_out = nn.Identity()

        def forward(self, in_) -> Tensor:
                # Main branch
                # Secondary branch
                b0 = self.block0(in_)
                b1 = self.block1(b0)
                b2 = self.block2(b1)
                b2 = self.attn(b2)
                do = self.do(b2)

                output = self.PReLU_out(self.conv_out(in_) + do)

                return output


class BottleNeckDownSampling(nn.Module):
        def __init__(self, in_dim, out_dim, projectionFactor):
                super().__init__()
                mid_dim: int = in_dim // projectionFactor

                # Main branch
                self.maxpool0 = nn.MaxPool2d(2, return_indices=True)

                # Secondary branch
                self.block0 = conv_block(in_dim, mid_dim, kernel_size=2, padding=0, stride=2)
                self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=1)
                self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)

                # Regularizer
                self.do = nn.Dropout(p=0.01)
                self.PReLU = nn.PReLU()

                # Out

        def forward(self, in_) -> tuple[Tensor, Tensor]:
                # Main branch
                maxpool_output, indices = self.maxpool0(in_)

                # Secondary branch
                b0 = self.block0(in_)
                b1 = self.block1(b0)
                b2 = self.block2(b1)
                do = self.do(b2)

                _, c, _, _ = maxpool_output.shape
                output = do
                output[:, :c, :, :] += maxpool_output

                final_output = self.PReLU(output)

                return final_output, indices


class BottleNeckUpSampling(nn.Module):
        def __init__(self, in_dim, out_dim, projectionFactor, up_mode="unpool"):
                super().__init__()
                mid_dim: int = in_dim // projectionFactor
                self.up_mode = up_mode

                # Main branch
                self.unpool = nn.MaxUnpool2d(2)

                # Secondary branch
                self.block0 = conv_block(in_dim, mid_dim, kernel_size=3, padding=1)
                self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=1)
                self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)

                # Regularizer
                self.do = nn.Dropout(p=0.01)
                self.PReLU = nn.PReLU()

                # Out

        def forward(self, args) -> Tensor:
                # nn.Sequential cannot handle multiple parameters:
                in_, indices, skip = args

                # Main branch
                if self.up_mode == "unpool":
                        up = self.unpool(in_, indices)
                elif self.up_mode == "resize":
                        up = F.interpolate(in_, scale_factor=2, mode="bilinear", align_corners=False)

                # Secondary branch
                b0 = self.block0(torch.cat((up, skip), dim=1))
                b1 = self.block1(b0)
                b2 = self.block2(b1)
                do = self.do(b2)

                output = self.PReLU(up + do)

                return output


class ENet(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, **kwargs):
                super().__init__()
                F: int = kwargs["factor"] if "factor" in kwargs else 4  # Projecting factor
                K: int = kwargs["kernels"] if "kernels" in kwargs else 16  # n_kernels

                # from models.enet import (BottleNeck,
                #                          BottleNeckDownSampling,
                #                          BottleNeckUpSampling,
                #                          conv_block)

                # Initial operations
                self.conv0 = nn.Conv2d(in_dim, K - 1, kernel_size=3, stride=2, padding=1)
                self.maxpool0 = nn.MaxPool2d(2, return_indices=False, ceil_mode=False)

                # Downsampling half
                self.bottleneck1_0 = BottleNeckDownSampling(K, K * 4, F)
                self.bottleneck1_1 = nn.Sequential(BottleNeck(K * 4, K * 4, F),
                                                   BottleNeck(K * 4, K * 4, F),
                                                   BottleNeck(K * 4, K * 4, F),
                                                   BottleNeck(K * 4, K * 4, F))
                self.bottleneck2_0 = BottleNeckDownSampling(K * 4, K * 8, F)

                # self.bottleneck2_1 = nn.Sequential(BottleNeck(K * 8, K * 8, F, dropoutRate=0.1),
                #                                    BottleNeck(K * 8, K * 8, F, dilation=2),
                #                                    BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True),
                #                                    BottleNeck(K * 8, K * 8, F, dilation=4),
                #                                    BottleNeck(K * 8, K * 8, F, dropoutRate=0.1),
                #                                    BottleNeck(K * 8, K * 8, F, dilation=8),
                #                                    BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True),
                #                                    BottleNeck(K * 8, K * 8, F, dilation=16))
                
                self.bottleneck2_1 = nn.Sequential(BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, attn="se", attn_red=16),
                                                   BottleNeck(K * 8, K * 8, F, dilation=2, attn="se", attn_red=16),
                                                   BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True, attn="se", attn_red=16),
                                                   BottleNeck(K * 8, K * 8, F, dilation=4, attn="se", attn_red=16),
                                                   BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, attn="se", attn_red=16),
                                                   BottleNeck(K * 8, K * 8, F, dilation=8, attn="se", attn_red=16),
                                                   BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True, attn="se", attn_red=16),
                                                   BottleNeck(K * 8, K * 8, F, dilation=16, attn="se", attn_red=16))

                # self.bottleneck2_1 = nn.Sequential(BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, c3=True),
                #                                    BottleNeck(K * 8, K * 8, F, dilation=2, c3=True),
                #                                    BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True, c3=True),
                #                                    BottleNeck(K * 8, K * 8, F, dilation=4, c3=True),
                #                                    BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, c3=True),
                #                                    BottleNeck(K * 8, K * 8, F, dilation=8, c3=True),
                #                                    BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True, c3=True),
                #                                    BottleNeck(K * 8, K * 8, F, dilation=16, c3=True))
                
                # self.bottleneck2_1 = nn.Sequential(BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, attn="se", attn_red=16, c3=True),
                #                                    BottleNeck(K * 8, K * 8, F, dilation=2, attn="se", attn_red=16, c3=True),
                #                                    BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True, attn="se", attn_red=16, c3=True),
                #                                    BottleNeck(K * 8, K * 8, F, dilation=4, attn="se", attn_red=16, c3=True),
                #                                    BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, attn="se", attn_red=16, c3=True),
                #                                    BottleNeck(K * 8, K * 8, F, dilation=8, attn="se", attn_red=16, c3=True),
                #                                    BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True, attn="se", attn_red=16, c3=True),
                #                                    BottleNeck(K * 8, K * 8, F, dilation=16, attn="se", attn_red=16, c3=True))

                # Middle operations
                # self.bottleneck3 = nn.Sequential(BottleNeck(K * 8, K * 8, F, dropoutRate=0.1),
                #                                  BottleNeck(K * 8, K * 8, F, dilation=2),
                #                                  BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True),
                #                                  BottleNeck(K * 8, K * 8, F, dilation=4),
                #                                  BottleNeck(K * 8, K * 8, F, dropoutRate=0.1),
                #                                  BottleNeck(K * 8, K * 8, F, dilation=8),
                #                                  BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True),
                #                                  BottleNeck(K * 8, K * 4, F, dilation=16, dilate_last=True))
                
                self.bottleneck3 = nn.Sequential(BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, attn="se", attn_red=16),
                                                 BottleNeck(K * 8, K * 8, F, dilation=2, attn="se", attn_red=16),
                                                 BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True, attn="se", attn_red=16),
                                                 BottleNeck(K * 8, K * 8, F, dilation=4, attn="se", attn_red=16),
                                                 BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, attn="se", attn_red=16),
                                                 BottleNeck(K * 8, K * 8, F, dilation=8, attn="se", attn_red=16),
                                                 BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True, attn="se", attn_red=16),
                                                 BottleNeck(K * 8, K * 4, F, dilation=16, dilate_last=True, attn="se", attn_red=16))
                
                # self.bottleneck3 = nn.Sequential(BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, c3=True),
                #                                  BottleNeck(K * 8, K * 8, F, dilation=2, c3=True),
                #                                  BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True, c3=True),
                #                                  BottleNeck(K * 8, K * 8, F, dilation=4, c3=True),
                #                                  BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, c3=True),
                #                                  BottleNeck(K * 8, K * 8, F, dilation=8, c3=True),
                #                                  BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True, c3=True),
                #                                  BottleNeck(K * 8, K * 4, F, dilation=16, dilate_last=True, c3=True))

                # self.bottleneck3 = nn.Sequential(BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, attn="se", attn_red=16, c3=True),
                #                                  BottleNeck(K * 8, K * 8, F, dilation=2, attn="se", attn_red=16, c3=True),
                #                                  BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True, attn="se", attn_red=16, c3=True),
                #                                  BottleNeck(K * 8, K * 8, F, dilation=4, attn="se", attn_red=16, c3=True),
                #                                  BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, attn="se", attn_red=16, c3=True),
                #                                  BottleNeck(K * 8, K * 8, F, dilation=8, attn="se", attn_red=16, c3=True),
                #                                  BottleNeck(K * 8, K * 8, F, dropoutRate=0.1, asym=True, attn="se", attn_red=16, c3=True),
                #                                  BottleNeck(K * 8, K * 4, F, dilation=16, dilate_last=True, attn="se", attn_red=16, c3=True))

                # Upsampling half
                self.bottleneck4 = nn.Sequential(BottleNeckUpSampling(K * 8, K * 4, F),
                                                 BottleNeck(K * 4, K * 4, F, dropoutRate=0.1),
                                                 BottleNeck(K * 4, K, F, dropoutRate=0.1))
                # self.bottleneck4 = nn.Sequential(BottleNeckUpSampling(K * 8, K * 4, F, up_mode="resize"),
                #                                  BottleNeck(K * 4, K * 4, F, dropoutRate=0.1),
                #                                  BottleNeck(K * 4, K, F, dropoutRate=0.1))
                
                self.bottleneck5 = nn.Sequential(BottleNeckUpSampling(K * 2, K, F),
                                                 BottleNeck(K, K, F, dropoutRate=0.1))
                # self.bottleneck5 = nn.Sequential(BottleNeckUpSampling(K * 2, K, F, up_mode="resize"),
                #                                  BottleNeck(K, K, F, dropoutRate=0.1))

                # Final upsampling and covolutions
                self.final = nn.Sequential(conv_block(K, K, kernel_size=3, padding=1, bias=False, stride=1),
                                           conv_block(K, K, kernel_size=3, padding=1, bias=False, stride=1),
                                           nn.Conv2d(K, out_dim, kernel_size=1))

                print(f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}) with {kwargs}")

        def forward(self, input):
                # Initial operations
                conv_0 = self.conv0(input)
                maxpool_0 = self.maxpool0(input)
                outputInitial = torch.cat((conv_0, maxpool_0), dim=1)

                # Downsampling half
                bn1_0, indices_1 = self.bottleneck1_0(outputInitial)
                bn1_out = self.bottleneck1_1(bn1_0)
                bn2_0, indices_2 = self.bottleneck2_0(bn1_out)
                bn2_out = self.bottleneck2_1(bn2_0)

                # Middle operations
                bn3_out = self.bottleneck3(bn2_out)

                # Upsampling half
                bn4_out = self.bottleneck4((bn3_out, indices_2, bn1_out))
                bn5_out = self.bottleneck5((bn4_out, indices_1, outputInitial))

                # Final upsampling and covolutions
                interpolated = F.interpolate(bn5_out, mode='nearest', scale_factor=2)
                # interpolated = F.interpolate(bn5_out, mode='bilinear', align_corners=False, scale_factor=2)
                return self.final(interpolated)

        def init_weights(self, *args, **kwargs):
                self.apply(random_weights_init)
