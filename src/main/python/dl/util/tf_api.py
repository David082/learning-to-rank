# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/6
version : 
"""
# ------ tf.nn.embedding_lookup函数的用法
# tf.nn.embedding_lookup（tensor, id）:tensor就是输入张量，id就是张量对应的索引
"""
import tensorflow as tf
import numpy as np

c = np.random.random([10,1])
b = tf.nn.embedding_lookup(c, [1, 3])

with tf.Session() as sess:
	# sess.run(tf.initialize_all_variables())
	print(sess.run(b))
	print(c)
"""
# ------ tensorflow-generative-model-collections
# https://github.com/hwalsuklee/tensorflow-generative-model-collections

# ------ tf.contrib.layers.variance_scaling_initializer
# 通过使用这种初始化方法，我们能够保证输入变量的变化尺度不变，从而避免变化尺度在最后一层网络中爆炸或者弥散。
# https://blog.csdn.net/u010185894/article/details/71104387
