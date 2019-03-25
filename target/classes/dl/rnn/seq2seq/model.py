# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/9
version : Minimal Seq2Seq model with Attention for Neural Machine Translation in PyTorch
refer : https://github.com/keon/seq2seq/blob/master/model.py
"""
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        embedded = self.embed(src)
        outputs, hidden = self.gru(embedded, hidden)
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:])
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size)
        self.v = nn.Parameter(torch.rand(out=self.hidden_size))


if __name__ == '__main__':
    rnn = nn.GRU(10, 20, 2)
    input = torch.randn(5, 3, 10)
    h0 = torch.randn(2, 3, 20)
    output, hn = rnn(input, h0)
