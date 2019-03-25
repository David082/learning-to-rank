# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2019/2/18
version :
refer :
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/autoencoder.ipynb
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data


class AutoEnCoder:
    def __init__(self, learning_rate=0.01, num_steps=30000, batch_size=256,
                 display_step=1000, examples_to_show=10,
                 num_hidden_1=256, num_hidden_2=128, num_input=784):
        # Training Parameters
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.batch_size = batch_size

        self.display_step = display_step
        self.examples_to_show = examples_to_show

        # Network Parameters
        self.num_hidden_1 = num_hidden_1  # 1st layer num features
        self.num_hidden_2 = num_hidden_2  # 2nd layer num features (the latent dim)
        self.num_input = num_input  # MNIST data input (img shape: 28*28)

    # Building the encoder
    def encoder(self, x, weights, biases):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.nn.xw_plus_b(x, weights['encoder_h1'], biases=biases['encoder_b1']),
                                name="encoder_layer1")
        # Encoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.nn.xw_plus_b(layer_1, weights['encoder_h2'], biases['encoder_b2']),
                                name="encoder_layer2")
        return layer_2

    # Building the decoder
    def decoder(self, x, weights, biases):
        # Decoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.nn.xw_plus_b(x, weights['decoder_h1'], biases['decoder_b1']), name="decoder_layer1")
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.nn.xw_plus_b(layer_1, weights['decoder_h2'], biases['decoder_b2']),
                                name="decoder_layer2")
        return layer_2

    def encoder_decoder_net(self):
        # tf Graph input (only pictures)
        X = tf.placeholder("float", [None, self.num_input], name="input")

        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.num_input, self.num_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_hidden_2])),
            'decoder_h1': tf.Variable(tf.random_normal([self.num_hidden_2, self.num_hidden_1])),
            'decoder_h2': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_input])),
        }
        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([self.num_hidden_2])),
            'decoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'decoder_b2': tf.Variable(tf.random_normal([self.num_input])),
        }

        # Construct model
        encoder_op = self.encoder(X, weights, biases)
        decoder_op = self.decoder(encoder_op, weights, biases)

        # Prediction
        y_pred = decoder_op

        # Targets (Labels) are the input data.
        y_true = X

        # Define loss and optimizer, minimize the squared error
        loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(loss)

        graph = {
            "X": X,
            "encoder_op": encoder_op,
            "y_pred": y_pred,
            "loss": loss,
            "optimizer": optimizer
        }

        return graph

    def train(self, sess, graph, mnist):
        # Training
        for i in range(1, self.num_steps + 1):
            # Prepare Data
            # Get the next batch of MNIST data (only images are needed, not labels)
            batch_x, _ = mnist.train.next_batch(self.batch_size)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, l = sess.run([graph['optimizer'], graph['loss']], feed_dict={graph['X']: batch_x})
            # Display logs per step
            if i % self.display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f' % (i, l))


if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST/", one_hot=True)
    # ------ https://stackoverflow.com/questions/49901806/tensorflow-importing-mnist-warnings
    # mnist = tf.keras.datasets.mnist
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    au = AutoEnCoder()

    # ------ Define graph
    tf.reset_default_graph()  # reset graph
    model = au.encoder_decoder_net()

    # ------ Training
    # Start Training
    # Start a new TF session
    sess = tf.Session()
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    # Run the initializer
    sess.run(init)
    au.train(sess, model, mnist)

    # ------ Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))

    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g = sess.run(model['y_pred'], feed_dict={model['X']: batch_x})  # predict

        # Display original images
        for j in range(n):
            # Draw the generated digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the generated digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()
