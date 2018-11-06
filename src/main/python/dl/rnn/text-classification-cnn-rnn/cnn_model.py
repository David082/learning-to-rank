# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/6
version :
refer :
CNN-RNN中文文本分类，基于TensorFlow
https://github.com/gaussic/text-classification-cnn-rnn
"""
import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""
    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 10  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5000  # 词汇表大小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config
        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, shape=[None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None, self.config.num_classes], name='input_x')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable(name='embedding', shape=[self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, ids=self.input_x)

        with tf.name_scope('cnn'):
            # CNN layer
            conv = tf.layers.conv1d(inputs=embedding_inputs, filters=self.config.num_filters,
                                    kernel_size=self.config.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(input_tensor=conv, reduction_indices=[1], name='gmp')

        with tf.name_scope('score'):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(inputs=gmp, units=self.config.hidden_dim, name='fc1')
            fc = tf.layers.dropout(inputs=fc, rate=self.keep_prob)
            fc = tf.nn.relu(features=fc)
            # 分类器
            self.logits = tf.layers.dense(inputs=fc, units=self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(input=tf.nn.softmax(logits=self.logits), axis=1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(input_tensor=cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(x=tf.argmax(input=self.input_y, axis=1), y=self.y_pred_cls)
            self.acc = tf.reduce_mean(input_tensor=tf.cast(x=correct_pred, dtype=tf.float32))
