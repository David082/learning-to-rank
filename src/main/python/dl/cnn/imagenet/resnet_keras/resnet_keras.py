# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/21
version :
refer :
https://github.com/mlhy/ResNet-50-for-Cats.Vs.Dogs/blob/master/ResNet-50%20for%20Cats.Vs.Dogs..ipynb
-- cv2: ImportError: DLL load failed
https://blog.csdn.net/cskywit/article/details/81513066


Build the structure of ResNet-50 for Cats.Vs.Dogs

    1.Define identity block.
    2.Define convolution block.
    3.Build the structure of ResNet-50 without top layer.
    4.Load weights
    5.Add top layer to ResNet-50.
    6.Setup training attribute.
    7.Compile the model.


"""
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, merge, Input
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D
from keras.utils import np_utils
from keras.models import model_from_json
from keras import backend as K
from keras.preprocessing import image
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
import random
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def data_preprocessing():
    """
    The images in train folder are divided into a training set and a validation set.
    The images both in training set and validation set are separately divided into two folders -- cat and dog according to their lables.

    (the two steps above were finished in Preprocessing train dataset.ipynb)

    The RGB color values of the images are rescaled to 0~1.
    The size of the images are resized to 224*224.

    """
    from keras.preprocessing.image import ImageDataGenerator
    image_width = 224
    image_height = 224
    image_size = (image_width, image_height)

    train_datagen = ImageDataGenerator(rescale=1.0 / 255)  # The RGB color values of the images are rescaled to 0~1.
    train_generator = train_datagen.flow_from_directory(
        PATH + 'mytrain',  # this is the target directory
        target_size=image_size,  # all images will be resized to 224x224
        batch_size=16,
        class_mode='binary')

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
    validation_generator = validation_datagen.flow_from_directory(
        PATH + 'myvalid',  # this is the target directory
        target_size=image_size,  # all images will be resized to 224x224
        batch_size=16,
        class_mode='binary')

    return train_generator, validation_generator


def show_images_randomly(train_generator):
    """show 16 images in the train dataset randomly
    """
    x, y = train_generator.next()
    plt.figure(figsize=(16, 8))
    for i, (img, label) in enumerate(zip(x, y)):
        plt.subplot(4, 4, i + 1)
        if label == 1:
            plt.title('dog')
        else:
            plt.title('cat')
        plt.axis('off')
        plt.imshow(img, interpolation='nearest')
    plt.show()


# 1. Define identity block.
def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity_block is the block that has no conv layer at shortcut.
    Arguments

        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, kernel_size=1, strides=1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same',)


if __name__ == '__main__':
    PATH = "C:/ImageNet/"
    train_generator, validation_generator = data_preprocessing()

    # show 16 images in the train dataset randomly
    show_images_randomly(train_generator)

    # 4.Load weights.
    TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/\
    v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # d:\\Users\\yu_wei\\.keras\\models\\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            TF_WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models',
                            md5_hash='a268eb855778b3df3c7506639542a6af')

    from tqdm import *

    test_num = 12500
    image_width = 224
    image_height = 224
    image_size = (image_width, image_height)
    image_matrix = np.zeros((test_num, image_width, image_height, 3), dtype=np.float32)


    def get_image(index):
        img = cv2.imread(PATH + 'test/%d.jpg' % index)
        img = cv2.resize(img, image_size)
        img.astype(np.float32)
        img = img / 255.0
        return img


    for i in tqdm(range(test_num)):
        image_matrix[i] = get_image(i + 1)
