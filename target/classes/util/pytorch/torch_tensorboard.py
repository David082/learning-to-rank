# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/30
version :
refer :
-- Pytorch使用Tensorboard可视化网络结构
https://blog.csdn.net/TTdreamloong/article/details/83107110

--
torch: 0.4.1
https://www.jianshu.com/p/bf3a46791f47
Invoked with: OperatorExportTypes.RAW, False
这个问题只出现在0.4.1, 将torch版本退回到0.4.0就ok了

-- tensorflow pretrained model
https://github.com/tensorflow/models/tree/master/research/slim
"""
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(),  # (6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),  # (16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


if __name__ == '__main__':
    dummy_input = Variable(torch.rand(13, 3, 299, 299))  # 假设输入13张1*28*28的图片
    # model = LeNet()

    from torchvision.models import AlexNet
    from torchvision.models import Inception3
    from torchvision.models import ResNet, DenseNet

    model = ResNet(Bottleneck, [3, 4, 6, 3])
    with SummaryWriter(comment='ResNet') as w:
        w.add_graph(model, (dummy_input,))

    from keras.applications import ResNet50
    import tensorflow as tf
    from tensorflow.contrib.slim import nets

    # model = ResNet50()
    model = torch.load("")



