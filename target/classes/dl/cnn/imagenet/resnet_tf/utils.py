# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/22
version :
refer :
https://github.com/jerryfan4/ResNet/blob/master/Utils.py
"""
import tensorflow as tf


def conv2d(scope, input_layer, output_dim, use_bias=False,
           filter_size=3, strides=[1, 1, 1, 1]):
    input_dim = input_layer.get_shape().as_list()[-1]

    with tf.variable_scope(scope):
        conv_filter = tf.get_variable(
            'conv_weight',
            shape=[filter_size, filter_size, input_dim, output_dim],
            dtype=tf.float32,
            initializer=tf.layers.variance_scaling_initializer(),
        )
