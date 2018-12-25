# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/12/17
version : 
"""
import torch
from torch import nn
from torch.autograd import Variable


class UnetSkipConnectBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)  # can optionally do the operation in-place
        downnorm = norm_layer(inner_nc)

        uprelu = nn.ReLU(inplace=True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], dim=1)


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                          innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                              norm_layer=norm_layer,
                                              use_dropout=use_dropout)
        unet_block = UnetSkipConnectBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                          norm_layer=norm_layer)
        self.model = unet_block

    def forward(self, input):
        return self.model(input)


class PerceptualLoss(nn.Module):
    def __init__(self, vgg_model):
        super(PerceptualLoss, self).__init__()
        self.vgg_model = vgg_model  # vgg_model为Imagenet上pre-trained vgg模型
        self.loss = nn.MSELoss()

    def forward(self, x, y):  # x为生成模型输出结果，y为label
        yc = Variable(y.data, volatitle=True)
        output_y = self.vgg_model(yc)
        output_x = self.vgg_model(x)[3]
        output_y_c = Variable(output_y[3].data, requires_grad=False)
        ploss = self.loss(output_x, output_y_c)
        return ploss
