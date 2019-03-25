# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2019/2/12
version :
refer :
-- keras lstm
https://github.com/DL-DeepLearning/Easy-deep-learning-with-Keras/blob/master/3.%20RNN/1-Basic-RNN/1-3-lstm.py
-- torch lstm
https://github.com/DL-DeepLearning/pytorch-tutorial/blob/master/tutorials/02-intermediate/recurrent_neural_network/main.py

reuters :
https://blog.csdn.net/cskywit/article/details/80900093
本数据库包含来自路透社的11,228条新闻，分为了46个主题。与IMDB库一样，每条新闻被编码为一个词下标的序列
"""
import numpy as np
from keras.datasets import reuters
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import accuracy_score

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import codecs
from torch.autograd import Variable

# parameters for data load
num_words = 30000
maxlen = 50
test_split = 0.3


def lstm():
    model = Sequential()
    model.add(LSTM(50, input_shape=(49, 1), return_sequences=False))
    model.add(Dense(46))
    model.add(Activation('softmax'))

    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def keras_lstm():
    (X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=num_words, maxlen=maxlen, test_split=test_split)

    # pad the sequences with zeros
    # padding parameter is set to 'post' => 0's are appended to end of sequences
    # post : pad after each sequence
    X_train = pad_sequences(X_train, padding='post')
    X_test = pad_sequences(X_test, padding='post')

    # reshape
    X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

    # one hot
    y_data = np.concatenate((y_train, y_test))
    y_data = to_categorical(y_data)  # 46个主题分类
    y_train = y_data[:1395]
    y_test = y_data[1395:]

    # model fitting
    # model = lstm()
    # model.fit(X_train, X_test, batch_size=50, epochs=200, verbose=1)

    model = KerasClassifier(build_fn=lstm, epochs=200, batch_size=50, verbose=1)
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)
    y_test_ = np.argmax(y_test, axis=1)

    print(accuracy_score(y_pred, y_test_))  # 0.8631051752921536


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)


class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'file:///E:/learning-to-rank/MNIST/train-images-idx3-ubyte.gz',
        'file:///E:/learning-to-rank/MNIST/train-labels-idx1-ubyte.gz',
        'file:///E:/learning-to-rank/MNIST/t10k-images-idx3-ubyte.gz',
        'file:///E:/learning-to-rank/MNIST/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        # from six.moves import urllib
        import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # batch_first : (batch, seq, feature)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out


if __name__ == '__main__':
    # Hyper Parameters
    sequence_length = 28
    input_size = 28
    hidden_size = 128
    num_layers = 2
    num_classes = 10
    batch_size = 100
    num_epochs = 2
    learning_rate = 0.01

    # MNIST Dataset
    train_dataset = MNIST(root='./MNIST/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

    test_dataset = MNIST(root='./MNIST/',
                         train=False,
                         transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    # torch RNN
    rnn = RNN(input_size, hidden_size, num_layers, num_classes)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    # Train the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, sequence_length, input_size))
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = rnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                # print(loss.item())  # loss.data[0] change to loss.item()
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))

    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, sequence_length, input_size))
        outputs = rnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

    # Save the Model
    torch.save(rnn, 'rnn.pth')

    # Load the model
    rnn = torch.load("rnn.pth")
