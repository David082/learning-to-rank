# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/12
version :
refer :
https://github.com/Ankur-Deka/Auto-Encoder/blob/master/utils.py
"""
import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets
import torch
from torch.utils.data import DataLoader


def load_test(path):
    if not os.path.exists(path):
        print('File {} doesn\'t exist'.format(path))
    else:
        img = Image.open(path)
        img = img.convert(mode='L')
        img = trans(img)
        x = img.view(1, 1, 60, 60)
        return (img.view(60, 60).numpy(), x)


if __name__ == '__main__':
    a = "hello"
    # root = './data'
    # if not os.path.exists(root):
    #     os.mkdir(root)
    #
    # batch_size = 100
    #
    # # if does not exist, download mnist dataset
    # trans = transforms.Compose([transforms.Resize(size=(60, 60)), transforms.ToTensor()])
    # train_set = datasets.MNIST(root=root, train=True, transform=trans, download=True)
    # valid_set = datasets.MNIST(root=root, train=False, transform=trans, download=True)
    #
    # train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    # valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True)
    #
    # # function to load images at test time
    # resize_trans = transforms.Compose([transforms.Resize(size=(60, 60)), transforms.ToTensor()])
