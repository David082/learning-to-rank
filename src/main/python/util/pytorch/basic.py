# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/27
version : 
"""
import torch
import numpy as np

if __name__ == '__main__':
    # numpy 和 torch.Tensor 之间的转换
    a = torch.rand(5, 4)
    b = a.numpy()
    print(b)
