# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/21
version :
refer :
https://github.com/sloth2012/AlexNet/blob/master/AlexNet.ipynb
"""
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d

model = Sequential(
    nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
)

model.parameters()

