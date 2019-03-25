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

# ------ batch_normalization / 退化学习率
# https://www.cnblogs.com/zyly/p/8996070.html
"""
# 加入退化学习率 初始值为learning_rate,让其每1000步，衰减0.9  学习率 = learning_rate*0.9^(global_step/1000)
global_step = tf.Variable(0,trainable=False)
decaylearning_rate = tf.train.exponential_decay(learning_rate,global_step,1000,0.9)
"""
# ------ tensorflow xgboost
# https://github.com/nicolov/gradient_boosting_tensorflow_xgboost
# https://stackoverflow.com/questions/53285128/export-tensorflow-graph-with-export-saved-model
# https://github.com/bartgras/XGBoost-Tensorflow-Wide-and-deep-comparison
"""
import tensorflow as tf
from tensorflow.contrib.boosted_trees.estimator_batch.estimator import GradientBoostedDecisionTreeClassifier
from tensorflow.contrib.boosted_trees.proto import learner_pb2

learner_config = learner_pb2.LearnerConfig()

# param
learning_rate = 0.1
l2 = 1.0
batch_size = 1024
depth = 6
examples_per_layer = 5000
num_trees = 10
feature_cols = None
output_dir = None

learner_config.learning_rate_tuner.fixed.learning_rate = learning_rate
learner_config.regularization.l1 = 0.0
# Set the regularization per instance in such a way that
# regularization for the full training data is equal to l2 flag.
learner_config.regularization.l2 = l2 / batch_size
learner_config.constraints.max_tree_depth = depth
learner_config.growing_mode = learner_pb2.LearnerConfig.LAYER_BY_LAYER

run_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=30)

# Create a TF Boosted trees regression estimator.
estimator = GradientBoostedDecisionTreeClassifier(
    learner_config=learner_config,
    examples_per_layer=examples_per_layer,
    n_classes=2,
    num_trees=num_trees,
    feature_columns=feature_cols,
    model_dir=output_dir,
    config=run_config,
    center_bias=False)
estimator.export_savedmodel()
"""
