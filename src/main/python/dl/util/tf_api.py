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
