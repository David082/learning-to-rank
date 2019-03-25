# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/26
version :
refer :
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
https://github.com/kratzert/finetune_alexnet_with_tensorflow/tree/5d751d62eb4d7149f4e3fd465febf8f07d4cea9d
"""
import os
import sys

sys.path.append("E:/learning-to-rank/src/main/python/dl/cnn/imagenet/alexnet_tf")

import numpy as np
import tensorflow as tf

# from alexnet import AlexNet
# from datagenerator import ImageDataGenerator
from dl.cnn.imagenet.alexnet_tf.alexnet import AlexNet
from dl.cnn.imagenet.alexnet_tf.datagenerator import ImageDataGenerator

from datetime import datetime

if __name__ == '__main__':
    """
    Configuration Part.
    """

    # Path to the textfiles for the trainings and validation set
    train_file = 'resources/cnn/alexnet/train.txt'
    val_file = 'resources/cnn/alexnet/val.txt'

    # Learning params
    learning_rate = 0.01
    num_epochs = 10
    batch_size = 128

    # Network params
    dropout_rate = 0.5
    num_classes = 2
    train_layers = ['fc8', 'fc7', 'fc6']

    # How often we want to write the tf.summary data to disk
    display_step = 20

    # Path for tf.summary.FileWriter and to store model checkpoints
    filewriter_path = "resources/cnn/alexnet/finetune_alexnet/tensorboard"
    checkpoint_path = "resources/cnn/alexnet/finetune_alexnet/checkpoints"

    """
    Main Part of the finetuning Script.
    """

    # Create parent path if it doesn't exist
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)

    # Place data loading and preprocessing on the cpu
    with tf.device('/cpu:0'):
        tr_data = ImageDataGenerator(train_file,
                                     mode='training',
                                     batch_size=batch_size,
                                     num_classes=num_classes,
                                     shuffle=True)
        val_data = ImageDataGenerator(val_file,
                                      mode='inference',
                                      batch_size=batch_size,
                                      num_classes=num_classes,
                                      shuffle=False)

        # create an reinitializable iterator given the dataset structure
        iterator = tf.data.Iterator.from_structure(tr_data.data.output_types,
                                                   tr_data.data.output_shapes)
        next_batch = iterator.get_next()
