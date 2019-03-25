# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/28
version :
refer :
https://github.com/amirhfarzaneh/vgg13-tensorlfow/blob/master/main.ipynb
"""
"""VGG13 CNN in TensorFlow
In this project we are going to:

1. Prepare the environment (e.g., importing the required libraries)
2. Load and preprocess a dataset to train on
3. Implement a fully functioning ConvNet using TensorFlow
4. Train the implemented model on the prepared dataset
5. Analyze the results
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


def vgg13(features, labels, mode):
    # Input Layer
    input_height, input_width = 28, 28
    input_channels = 1
    input_layer = tf.reshape(features["x"], [-1, input_height, input_width, input_channels])

    # Convolutional Layer #1
    conv1_1 = tf.layers.conv2d(input_layer, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    conv1_2 = tf.layers.conv2d(conv1_1, filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv1_2, pool_size=[2, 2], strides=(2, 2), padding='same')  # Pooling Layer #2

    # Convolutional Layer #3
    conv3_1 = tf.layers.conv2d(pool2, filters=128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    conv3_2 = tf.layers.conv2d(conv3_1, filters=128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    pool4 = tf.layers.max_pooling2d(conv3_2, pool_size=[2, 2], strides=(2, 2), padding='same')  # Pooling Layer #4

    # Convolutional Layer #5
    conv5_1 = tf.layers.conv2d(pool4, filters=256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    conv5_2 = tf.layers.conv2d(conv5_1, filters=256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    pool6 = tf.layers.max_pooling2d(conv5_2, pool_size=[2, 2], strides=(2, 2), padding='same')  # Pooling Layer #6

    # Convolutional Layer #7
    conv7_1 = tf.layers.conv2d(pool6, filters=512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    conv7_2 = tf.layers.conv2d(conv7_1, filters=512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    pool8 = tf.layers.max_pooling2d(conv7_2, pool_size=[2, 2], strides=(2, 2), padding='same')  # Pooling Layer #8

    # Convolutional Layer #9
    conv9_1 = tf.layers.conv2d(pool8, filters=512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    conv9_2 = tf.layers.conv2d(conv9_1, filters=512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
    pool10 = tf.layers.max_pooling2d(conv9_2, pool_size=[2, 2], strides=(2, 2), padding='same')  # Pooling Layer #10

    # FC Layers
    pool10_flat = tf.layers.flatten(inputs=pool10)
    FC11 = tf.layers.dense(inputs=pool10_flat, units=4096, activation=tf.nn.relu)  # FC Layers # 11
    FC12 = tf.layers.dense(inputs=FC11, units=4096, activation=tf.nn.relu)  # FC Layers # 12
    FC13 = tf.layers.dense(inputs=FC12, units=1000, activation=tf.nn.relu)  # FC Layers # 13

    """the training argument takes a boolean specifying whether or not the model is currently
    being run in training mode; dropout will only be performed if training is true. here,
    we check if the mode passed to our model function cnn_model_fn is train mode. """
    dropout = tf.layers.dropout(inputs=FC13, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer or the output layer. which will return the raw values for our predictions.
    # Like FC layer, logits layer is another dense layer. We leave the activation function empty
    # so we can apply the softmax
    logits = tf.layers.dense(inputs=dropout, units=10)

    # Then we make predictions based on raw output
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        # the predicted class for each example - a vlaue from 0-9
        "classes": tf.argmax(input=logits, axis=1),
        # to calculate the probablities for each target class we use the softmax
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # so now our predictions are compiled in a dict object in python and using that we return an estimator object
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    '''Calculate Loss (for both TRAIN and EVAL modes): computes the softmax entropy loss.
    This function both computes the softmax activation function as well as the resulting loss.'''
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Options (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)


if __name__ == '__main__':
    # 2. Load and preprocess the dataset
    # Loading the data (signs)
    mnist = input_data.read_data_sets("MNIST/")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Example of a picture
    train_data.shape
    index = 7
    plt.imshow(train_data[index].reshape(28, 28))
    plt.show()
    print("y = " + str(np.squeeze(train_labels[index])))

    # get some statistic from the dataset
    print("number of training examples = " + str(train_data.shape[0]))
    print("number of evaluation examples = " + str(eval_data.shape[0]))
    print("X_train shape: " + str(train_data.shape))
    print("Y_train shape: " + str(train_labels.shape))
    print("X_test shape: " + str(eval_data.shape))
    print("Y_test shape: " + str(eval_labels.shape))

    # ------ Training
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=vgg13,
                                              model_dir="models/mnist_vgg13_model")

    # Train the model
    with tf.device("/cpu:0"):
        train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},
                                                            y=train_labels,
                                                            batch_size=100,
                                                            num_epochs=100,
                                                            shuffle=True)
        mnist_classifier.train(input_fn=train_input_fn,
                               steps=None,
                               hooks=None)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},
                                                       y=eval_labels,
                                                       num_epochs=1,
                                                       shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
