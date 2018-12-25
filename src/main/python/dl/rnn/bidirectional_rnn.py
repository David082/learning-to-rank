# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/12/5
version :
refer :
https://github.com/fuqiuai/TensorFlow-Deep-Learning/blob/master/TensorFlow_BiRNN%2CBiLSTM%2CBiGRU.ipynb
"""
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def BiRNN(x, weights, bias):
    """定义BiRNN网络
    """
    '''返回[batch_size, n_classes]'''
    x = tf.reshape(x, shape=[-1, sequence_length, frame_size])
    fw_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_num)
    bw_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_num)

    # BiRNN/BiLSTM/BiGRU
    # 输出(outputs, states为)
    # outputs为(output_fw, output_bw), states为(output_state_fw, output_state_bw)
    outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)
    output = tf.concat(outputs, 2)

    return tf.nn.softmax(tf.matmul(output[:, -1, :], weights) + bias, 1)


if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST/", one_hot=True)
    print(mnist.train.images.shape, mnist.train.labels.shape)
    print(mnist.test.images.shape, mnist.test.labels.shape)

    train_rate = 0.001  # 学习速率
    train_step = 1000
    batch_size = 1280  # 每批样本数
    display_step = 100  # 控制输出频次

    frame_size = 28  # 序列里面每一个分量的大小。因为每个分量都是一行像素，而一行像素有28个像素点。所以frame_size为28
    sequence_length = 28  # 每个样本序列的长度。因为我们希望把一个28x28的图片当做一个序列输入到rnn进行训练，所以我们需要对图片进行序列化。一种最方便的方法就是我们认为行与行之间存在某些关系，于是把图片的每一行取出来当做序列的一个维度。所以这里sequence_size就是设置为28。
    hidden_num = 100  # 隐层个数
    n_classes = 10  # 类别数

    # 定义输入,输出
    x = tf.placeholder(dtype=tf.float32, shape=[None, sequence_length * frame_size], name="input_x")
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name="expected_y")
    # 定义权值
    weights = tf.Variable(tf.truncated_normal(shape=[hidden_num * 2, n_classes]))
    bias = tf.Variable(tf.zeros(shape=[n_classes]))

    # 计算预计输出
    predy = BiRNN(x, weights, bias)
    # 定义损失函数和优化算法
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predy, labels=y))
    train = tf.train.AdamOptimizer(train_rate).minimize(cost)
    # 计算accuracy
    correct_pred = tf.equal(tf.argmax(predy, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.to_float(correct_pred))

    # 开始训练
    with tf.Session() as sess:
        print('step', 'accuracy', 'loss')
        sess.run(tf.global_variables_initializer())
        step = 1
        testx, testy = mnist.test.next_batch(batch_size)
        while step < train_step:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # batch_x = tf.reshape(batch_x, shape=[batch_size, sequence_length, frame_size])
            _loss, __ = sess.run([cost, train], feed_dict={x: batch_x, y: batch_y})
            if step % display_step == 0:
                acc, loss = sess.run([accuracy, cost], feed_dict={x: testx, y: testy})
                print(step, acc, loss)
            step += 1
    # Save log for tensorboard
    tf.summary.FileWriter("logs", sess.graph)
