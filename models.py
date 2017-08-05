# -*- coding: utf-8 -*-
"""Implements SRGAN models: https://arxiv.org/abs/1609.04802

TODO:
    * Try to make this work with SELU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=8):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)

class denseNet(nn.Module):
    def __init__(self, in_channels, k, layers, p=0.2):
        super(denseNet, self).__init__()
        self.layers = layers

        for i in range(layers):
            self.add_module('batchnorm' + str(i+1), nn.BatchNorm2d(in_channels))
            self.add_module('conv' + str(i+1), nn.Conv2d(in_channels, k, 3, stride=1, padding=1))
            self.add_module('drop' + str(i+1), nn.Dropout2d(p=p))
            in_channels += k

    def forward(self, x):
        for i in range(self.layers):
            y = self.__getattr__('batchnorm' + str(i+1))(x.clone())
            y = F.elu(y)
            y = self.__getattr__('conv' + str(i+1))(y)
            y = self.__getattr__('drop' + str(i+1))(y)
            x = torch.cat((x,y), dim=1)
        return x

class upsampleBlock(nn.Module):
    # Implements resize-convolution
    def __init__(self, in_channels, out_channels):
        super(upsampleBlock, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        return F.elu(self.conv1(self.upsample1(x)))

class Generator(nn.Module):
    def __init__(self, n_dense_blocks, upsample):
        super(Generator, self).__init__()
        self.n_dense_blocks = n_dense_blocks
        self.upsample = upsample

        self.conv1 = nn.Conv2d(3, 64, 9, stride=1, padding=1)

        inchannels = 64
        for i in range(self.n_dense_blocks):
            self.add_module('denseNet' + str(i+1), denseNet(inchannels, 12, 4))
            inchannels += 12*4

        self.conv2 = nn.Conv2d(inchannels, 64, 3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)

        in_channels = 64
        out_channels = 256
        for i in range(self.upsample):
            self.add_module('upsample' + str(i+1), upsampleBlock(in_channels, out_channels))
            in_channels = out_channels
            out_channels = out_channels/2

        self.conv3 = nn.Conv2d(in_channels, 3, 9, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)

        for i in range(self.n_dense_blocks):
            x = self.__getattr__('denseNet' + str(i+1))(x)

        x = F.elu(self.conv2_bn(self.conv2(x)))

        for i in range(self.upsample):
            x = self.__getattr__('upsample' + str(i+1))(x)

        return self.conv3(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.conv6_bn = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv7_bn = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.conv8_bn = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.elu(self.conv1(x))

        x = F.elu(self.conv2_bn(self.conv2(x)))
        x = F.elu(self.conv3_bn(self.conv3(x)))
        x = F.elu(self.conv4_bn(self.conv4(x)))
        x = F.elu(self.conv5_bn(self.conv5(x)))
        x = F.elu(self.conv6_bn(self.conv6(x)))
        x = F.elu(self.conv7_bn(self.conv7(x)))
        x = F.elu(self.conv8_bn(self.conv8(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        x = F.elu(self.fc1(x))
        return F.sigmoid(self.fc2(x))
