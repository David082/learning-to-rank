# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/28
version :
refer :
https://github.com/transcranial/wide-resnet/blob/master/wide-resnet.ipynb
"""
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import Dense, Activation, Flatten, Lambda
from keras.layers import Convolution2D, AveragePooling2D, BatchNormalization, Dropout
from keras.layers import merge, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import keras.backend as K
import json
import time


def zero_pad_channels(x, pad=0):
    """Function for Lambda layer
    """
    patten = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
    return tf.pad(tensor=x, paddings=patten)


def residual_block(x, nb_filters=16, subsample_factor=1):
    prev_nb_channesl = K.int_shape(x)[3]
    if subsample_factor > 1:
        subsample = (subsample_factor, subsample_factor)
        # shortcut: subsample + zero-pad channel dim
        shorcut = AveragePooling2D(pool_size=subsample, dim_ordering='tf')(x)
    else:
        subsample = (1, 1)
        # shortcut: identity
        shorcut = x

    if nb_filters > prev_nb_channesl:
        shorcut = Lambda(zero_pad_channels, arguments={'pad': nb_filters - prev_nb_channesl})(shorcut)

    y = BatchNormalization(axis=3)(x)
    y = Activation('relu')(y)
    y = Convolution2D(filters=nb_filters, kernel_size=3, strides=3,)

if __name__ == '__main__':
    nb_classes = 10

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # reorder dimensions for tensorflow
    # x_train = np.transpose(x_train.astype('float32'), (0, 2, 3, 1))
    # x_test = np.transpose(x_test.astype('float32'), (0, 2, 3, 1))
    mean = np.mean(x_train, axis=0, keepdims=True)
    std = np.std(x_train)

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # wide residual network
