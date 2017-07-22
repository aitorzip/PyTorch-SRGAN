# -*- coding: utf-8 -*-
"""Implements SRGAN models: https://arxiv.org/abs/1609.04802

Todo:
"""

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# custom weights initialization for SELU activations
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_in = size[1]
        nn.init.normal(m.weight.data, mean=0, std=np.sqrt(1.0/fan_in))
        nn.init.constant(m.bias.data, 0.0)

class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:feature_layer])

    def forward(self, x):
        return self.features(x)

class residualBlock(nn.Module):
    def __init__(self, in_channels):
        super(residualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv2(F.selu(self.conv1(x))) + x

class upsampleBlock(nn.Module):
    # Implements resize-convolution
    def __init__(self, in_channels, out_channels):
        super(upsampleBlock, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        return F.selu(self.conv1(self.upsample1(x)))

class Generator(nn.Module):
    def __init__(self, n_residual_blocks, upsample):
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample = upsample

        self.conv1 = nn.Conv2d(3, 64, 9, stride=1, padding=1)

        for i in range(self.n_residual_blocks):
            self.add_module('res' + str(i+1), residualBlock(64))

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        in_channels = 64
        out_channels = 256
        for i in range(self.upsample):
            self.add_module('upscale' + str(i+1), upsampleBlock(in_channels, out_channels))
            in_channels = out_channels
            out_channels = out_channels/2

        self.conv3 = nn.Conv2d(in_channels, 3, 9, stride=1, padding=1)

    def forward(self, x):
        x = F.selu(self.conv1(x))

        y = self.__getattr__('res1')(x)
        for i in range(1, self.n_residual_blocks):
            y = self.__getattr__('res' + str(i+1))(y)

        x = self.conv2(y) + x

        for i in range(self.upsample):
            x = self.__getattr__('upscale' + str(i+1))(x)

        return F.sigmoid(self.conv3(x))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.selu(self.conv1(x))

        x = F.selu(self.conv2(x))
        x = F.selu(self.conv3(x))
        x = F.selu(self.conv4(x))
        x = F.selu(self.conv5(x))
        x = F.selu(self.conv6(x))
        x = F.selu(self.conv7(x))
        x = F.selu(self.conv8(x))

        # Flatten
        x = x.view(x.size(0), -1)

        x = F.selu(self.fc1(x))
        return F.sigmoid(self.fc2(x))
