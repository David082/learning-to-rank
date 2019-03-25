# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/28
version :
refer :
https://github.com/bentrevett/pytorch-image-classification/blob/master/6%20-%20ResNet.ipynb
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import random
import numpy as np


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNetLayer(nn.Module):
    def __init__(self, block, n_blocks, in_channels, out_channels, stride):
        super().__init__()

        self.modules = []

        self.modules.append(block(in_channels, out_channels, stride))

        for _ in range(n_blocks - 1):
            self.modules.append(block(out_channels, out_channels, 1))

        self.blocks = nn.Sequential(*self.modules)

    def forward(self, x):
        return self.blocks(x)


class ResNet18(nn.Module):
    def __init__(self, layer, block):
        super().__init__()

        n_blocks = [2, 2, 2, 2]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = layer(block, n_blocks[0], 64, 64, 1)
        self.layer2 = layer(block, n_blocks[1], 64, 128, 2)
        self.layer3 = layer(block, n_blocks[2], 128, 256, 2)
        self.layer4 = layer(block, n_blocks[3], 256, 512, 2)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out


class ResNet34(nn.Module):
    def __init__(self, layer, block):
        super().__init__()

        n_blocks = [3, 4, 6, 3]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = layer(block, n_blocks[0], 64, 64, 1)
        self.layer2 = layer(block, n_blocks[1], 64, 128, 2)
        self.layer3 = layer(block, n_blocks[2], 128, 256, 2)
        self.layer4 = layer(block, n_blocks[3], 256, 512, 2)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    device = torch.device('cuda')
    model = ResNet18(ResNetLayer, ResNetBlock).to(device)
    print(model)

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomCrop(32, padding=3),
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
    ])

    train_data = datasets.CIFAR10('data', train=True, download=True, transform=train_transforms)
