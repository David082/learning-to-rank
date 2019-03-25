# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2019/3/6
version :
refer :
https://github.com/DL-DeepLearning/pytorch-tutorial/tree/master/tutorials/02-intermediate
"""
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_layers=2, num_classes=10,
                 sequence_length=28):
        super(RNN, self).__init__()
        # Hyper Parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.sequence_length = sequence_length
        # self.batch_size = batch_size
        # self.num_epochs = num_epochs
        # self.learning_rate = learning_rate

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # FC layer
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


def train(rnn, num_epochs, batch_size, learning_rate):
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    # Train the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, rnn.sequence_length, rnn.input_size))
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = rnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                      # % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))


def test(rnn):
    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, rnn.sequence_length, rnn.input_size))
        outputs = rnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
    correct += (predicted == labels).sum()
    print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    # Hyper Parameters
    batch_size = 100
    num_epochs = 2
    learning_rate = 0.01
    # model
    rnn = RNN()
    # MNIST Dataset
    train_dataset = dsets.MNIST(root='E:/learning-to-rank/MNIST/',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=False)

    test_dataset = dsets.MNIST(root='E:/learning-to-rank/MNIST/',
                               train=False,
                               transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    # Training
    train(rnn, num_epochs, batch_size, learning_rate)

    # # Save the Model
    # torch.save(rnn.state_dict(), 'rnn.pkl')
