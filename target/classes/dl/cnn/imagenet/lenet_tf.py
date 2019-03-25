# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/21
version :
refer :
https://github.com/sujaybabruwad/LeNet-in-Tensorflow/blob/master/LeNet-Lab.ipynb
https://blog.csdn.net/qq_33039859/article/details/79907879

ImageNet
https://blog.csdn.net/Bruce_0712/article/details/80287467
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
from tensorflow.python.framework import graph_util
import numpy as np
import matplotlib.pyplot as plt


class LeNet(object):
    def __init__(self, epochs=10, batch_size=128, mu=0, sigma=0.1,
                 learning_rate=0.001,
                 num_class=10,
                 pb_file_path=None,
                 saver=None):
        # Hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.mu = mu
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.num_class = num_class
        # Saver
        self.pb_file_path = pb_file_path
        self.saver = saver

    def load_data(self):
        mnist = input_data.read_data_sets("MNIST/", reshape=False)
        X_train, y_train = mnist.train.images, mnist.train.labels
        X_validation, y_validation = mnist.validation.images, mnist.validation.labels
        X_test, y_test = mnist.test.images, mnist.test.labels

        assert (len(X_train) == len(y_train))
        assert (len(X_validation) == len(y_validation))
        assert (len(X_test) == len(y_test))

        # Pad images with 0s
        X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
        X_validation = np.pad(X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
        X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

        print("Updated Image Shape: {}".format(X_train[0].shape))

        return X_train, y_train, X_validation, y_validation, X_test, y_test

    def net(self):
        # Features and Labels
        x = tf.placeholder(tf.float32, (None, 32, 32, 1), name="input")
        y = tf.placeholder(tf.int32, (None), name="y")
        one_hot_y = tf.one_hot(y, self.num_class, name="labels")

        # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        # tf.layers.conv2d()
        conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=self.mu, stddev=self.sigma))
        conv1_b = tf.Variable(tf.zeros(6))
        conv1 = tf.nn.conv2d(input=x, filter=conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
        # TODO: Activation.
        conv1 = tf.nn.relu(conv1)

        # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
        pool_1 = tf.nn.max_pool(value=conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # TODO: Layer 2: Convolutional. Output = 10x10x16.
        conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=self.mu, stddev=self.sigma))
        conv2_b = tf.Variable(tf.zeros(16))
        conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
        # TODO: Activation.
        conv2 = tf.nn.relu(conv2)

        # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
        pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # TODO: Flatten. Input = 5x5x16. Output = 400.
        fc1 = flatten(pool_2)

        # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
        fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=self.mu, stddev=self.sigma))
        fc1_b = tf.Variable(tf.zeros(120))
        fc1 = tf.matmul(fc1, fc1_w) + fc1_b
        # TODO: Activation.
        fc1 = tf.nn.relu(fc1)

        # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=self.mu, stddev=self.sigma))
        fc2_b = tf.Variable(tf.zeros(84))
        fc2 = tf.matmul(fc1, fc2_w) + fc2_b
        # TODO: Activation.
        fc2 = tf.nn.relu(fc2)

        # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
        fc3_w = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=self.mu, stddev=self.sigma))
        fc3_b = tf.Variable(tf.zeros(10))
        # logits = tf.matmul(fc2, fc3_w) + fc3_b
        logits = tf.nn.xw_plus_b(fc2, fc3_w, fc3_b, name="output")

        # TODO: model eval
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
        loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        graph = {
            "x": x,
            "y": y,
            "logits": logits,
            "optimizer": optimizer,
            "accuracy": accuracy
        }

        return graph

    def evaluate(self, graph, x, y):
        num_examples = len(x)
        total_accuracy = 0
        sess = tf.get_default_session()  # get_default_session
        for offset in range(0, num_examples, self.batch_size):
            batch_x, batch_y = x[offset:offset + self.batch_size], y[offset:offset + self.batch_size]
            accuracy = sess.run(graph['accuracy'], feed_dict={graph['x']: batch_x, graph['y']: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    def train(self, graph, train_x, train_y, valid_x, valid_y):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            num_examples = len(train_x)

            print("Training...")
            print()
            for i in range(self.epochs):
                train_x, train_y = shuffle(train_x, train_y)
                # total_accuracy = 0

                for offset in range(0, num_examples, self.batch_size):
                    end = offset + self.batch_size
                    batch_x, batch_y = train_x[offset:end], train_y[offset:end]
                    _, accuracy = sess.run([graph["optimizer"], graph["accuracy"]],
                                           feed_dict={graph['x']: batch_x, graph['y']: batch_y})

                validation_accuracy = self.evaluate(graph, valid_x, valid_y)
                print("EPOCH {} ...".format(i + 1))
                print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                print()

            # self.saver.save(sess, 'lenet')
            print("Model saved")
            # Save model to pb file
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
            with tf.gfile.FastGFile(self.pb_file_path, mode='wb') as f:
                f.write(constant_graph.SerializeToString())
            # Save log for tensorboard
            tf.summary.FileWriter("logs", sess.graph)

    def model_eval(self, pb_file_path, test_x, test_y):
        with tf.gfile.FastGFile(pb_file_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        # graph weight and bias
        with tf.Session() as sess:
            x = sess.graph.get_tensor_by_name("input:0")
            y = tf.placeholder(tf.int32, (None))
            one_hot_y = tf.one_hot(y, self.num_class)

            logits = sess.graph.get_tensor_by_name("output:0")
            # predict
            predict = sess.run(logits, feed_dict={x: test_x})

            # eval
            correct = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

            test_accuracy = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
            print("Test Accuracy = {:.3f}".format(test_accuracy))

        return predict

    def saver_eval(self, graph, test_x, test_y):
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint("."))
            test_accuracy = self.evaluate(graph, test_x, test_y)
            print("Test Accuracy = {:.3f}".format(test_accuracy))


def plot_image(x):
    # image = x.squeeze()
    # plt.figure(figsize=(1, 1))
    # plt.imshow(image, cmap="gray")
    # plt.show()
    r, c = 5, 5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(x[cnt, :, :, 0].squeeze(), cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    plt.show()


if __name__ == '__main__':
    pb_file_path = "models/lenet.pb"
    lenet = LeNet(pb_file_path=pb_file_path)
    X_train, y_train, X_validation, y_validation, X_test, y_test = lenet.load_data()

    # training
    tf.reset_default_graph()  # Clears the default graph stack and resets the global default graph.
    graph = lenet.net()
    lenet.train(graph, X_train, y_train, X_validation, y_validation)

    # eval
    predict = lenet.model_eval(pb_file_path, X_test, y_test)
    plot_image(X_test)
