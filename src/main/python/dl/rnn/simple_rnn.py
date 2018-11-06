# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/6
version :
refer : https://github.com/DL-DeepLearning/deep-learning-keras-tensorflow/tree/master/7.%20Recurrent%20Neural%20Networks
"""
from keras.datasets import imdb
from keras.layers import Input, Embedding, SimpleRNN, Dropout, Activation
from keras.models import Model


class RNNExample:
    def __init__(self, batch_size=32, max_len=100, max_features=20000):
        self.batch_size = batch_size
        self.max_len = max_len
        self.max_features = max_features

    def build_network(self):
        # network
        input = Input(shape=(self.max_len,), name="input")
        x = Embedding(input_dim=self.max_features, output_dim=128, input_length=self.max_len)(input)
        x = SimpleRNN(units=128)(x)
        x = Dropout(rate=0.5)(x)
        output = Activation(activation='sigmoid', name='output')(x)
        # model
        model = Model(inputs=input, outputs=output)
        # try using different optimizers and different optimizer configs
        model.compile(loss='binary_crossentropy', optimizer='adam')

        return model


if __name__ == '__main__':
    max_features = 20000
    maxlen = 100  # cut texts after this number of words (among top max_features most common words)

    print("Loading data...")
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    rnn_e = RNNExample()
    # build model
    model = rnn_e.build_network()

    print("Train...")
    model.fit(X_train, y_train, batch_size=rnn_e.batch_size, epochs=1,
              validation_data=(X_test, y_test))
