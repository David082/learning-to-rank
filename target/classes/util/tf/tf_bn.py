# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/27
version :
refer :
https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-13-BN/
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def built_net(xs, ys, norm):
    def add_layer(inputs, in_size, out_size, activation_function=None):
        # 添加层功能
        Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0., stddev=1.))
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

    # fix_seed(1)

    layers_inputs = [xs]  # 记录每层的 input

    # loop 建立所有层
    for l_n in range(N_LAYERS):
        layer_input = layers_inputs[l_n]
        in_size = layers_inputs[l_n].get_shape()[1].value

        output = add_layer(
            layer_input,  # input
            in_size,  # input size
            N_HIDDEN_UNITS,  # output size
            ACTIVATION,  # activation function
        )
        layers_inputs.append(output)  # 把 output 加入记录

    # 建立 output layer
    prediction = add_layer(layers_inputs[-1], 30, 1, activation_function=None)

    cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    return [train_op, cost, layers_inputs]


if __name__ == '__main__':
    ACTIVATION = tf.nn.relu  # 每一层都使用 relu
    N_LAYERS = 7  # 一共7层隐藏层
    N_HIDDEN_UNITS = 30  # 每个层隐藏层有 30 个神经元

    x_data = np.linspace(-7, 10, 500)[:, np.newaxis]
    noise = np.random.normal(0, 8, x_data.shape)
    y_data = np.square(x_data) - 5 + noise

    # 可视化 input data
    plt.scatter(x_data, y_data)
    plt.show()
