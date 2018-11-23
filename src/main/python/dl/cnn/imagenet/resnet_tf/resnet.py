# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/22
version :
refer :
https://github.com/jerryfan4/ResNet
"""
from dl.cnn.imagenet.resnet_tf.utils import *
import tensorflow as tf


def residual(scope, input_layer, is_training, reuse,
             increase_dim=False, first=False):
    input_dim = input_layer.get_shape().as_list()[-1]

    if increase_dim:
        output_dim = input_dim * 2
        strides = [1, 2, 2, 1]
    else:
        output_dim = input_dim
        strides = [1, 1, 1, 1]

    with tf.variable_scope(scope):
        if first:
            h0 = input_layer


if __name__ == '__main__':
    print("test")
