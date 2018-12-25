# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/30
version :
refer :
-- Pytorch 模型的网络结构可视化
https://blog.csdn.net/TTdreamloong/article/details/83107110
"""
import torch
from torch import nn
import sys

sys.path.append("E:/learning-to-rank/src/main/python/util/pytorch")

# from util.pytorch.dot import make_dot
from dot import make_dot
from torch.autograd import Variable


def print_model_params(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        print("layer shape:" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("layer total params: " + str(l))
        k = k + l
    print("sum of params: " + str(k))


if __name__ == '__main__':
    # MLP
    model = nn.Sequential()
    model.add_module(name="W0", module=nn.Linear(in_features=8, out_features=16))
    model.add_module(name="tanh", module=nn.Tanh())
    model.add_module(name="W1", module=nn.Linear(in_features=16, out_features=1))

    x = Variable(torch.randn(1, 8))

    vis_graph = make_dot(var=model(x), params=dict(model.named_parameters()))
    vis_graph.view()

    # AlexNet
    from torchvision.models import AlexNet

    model = AlexNet()

    x = torch.randn(1, 3, 227, 227).requires_grad_(True)
    y = model(x)
    vis_graph = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
    vis_graph.view()

    # print params
    print_model_params(model)
