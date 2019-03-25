# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2019/2/28
version :
refer :
https://github.com/keon/seq2seq/blob/master/model.py
"""
import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import re
import spacy
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
from torch import optim
from torch.nn.utils import clip_grad_norm


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(num_embeddings=input_size, embedding_dim=embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=n_layers, dropout=dropout, bidirectional=True)

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
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))  # A kind of Tensor that is to be considered a module parameter
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(hidden=h, encoder_outputs=encoder_outputs)
        # https://github.com/keon/seq2seq/issues/12
        # return F.relu(attn_energies, dim=1).unsqueeze(1)
        return F.relu(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        # energy = F.softmax(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = F.softmax(self.attn(torch.cat([hidden, encoder_outputs], 2)), dim=2)
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(num_embeddings=output_size, embedding_dim=embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], dim=2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, cuda_available=False):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cuda_available = cuda_available

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        if self.cuda_available:
            outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()
        else:
            outputs = Variable(torch.zeros(max_len, batch_size, vocab_size))

        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]
        output = Variable(trg.data[0, :])
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(output, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            if self.cuda_available:
                output = Variable(trg.data[t] if is_teacher else top1).cuda()
            else:
                output = Variable(trg.data[t] if is_teacher else top1)
        return outputs


def load_dataset(batch_size):
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')
    url = re.compile('(<url>.*</url>)')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]

    DE = Field(tokenize=tokenize_de, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    EN = Field(tokenize=tokenize_en, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    train, val, test = Multi30k.splits(exts=('.de', '.en'), fields=(DE, EN))
    DE.build_vocab(train.src, min_freq=2)
    EN.build_vocab(train.trg, max_size=10000)
    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train, val, test), batch_size=batch_size, repeat=False)
    return train_iter, val_iter, test_iter, DE, EN


def train(e, model, optimizer, train_iter, vocab_size, grad_clip, DE, EN):
    model.train()
    total_loss = 0
    pad = EN.vocab.stoi['<pad>']
    for b, batch in enumerate(train_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        if cuda_available:
            src, trg = src.cuda(), trg.cuda()
        optimizer.zero_grad()
        output = model(src, trg)
        loss = F.nll_loss(input=output[1:].view(-1, vocab_size),
                          target=trg[1:].contiguous().view(-1),
                          ignore_index=pad)
        loss.backward()
        clip_grad_norm(parameters=model.parameters(), max_norm=grad_clip)
        optimizer.step()
        # total_loss += loss.data[0]
        total_loss += loss.item()

        if b % 100 == 0 and b != 0:
            total_loss /= 100
            print("[%d][loss:%5.2f][pp:%5.2f]" % (b, total_loss, math.exp(total_loss)))
            total_loss = 0


def evaluate(model, val_iter, vocab_size, DE, EN):
    model.eval()
    pad = EN.vocab.stoi['<pad>']
    total_loss = 0
    for b, batch in enumerate(val_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        if cuda_available:
            src = Variable(src.data.cuda(), volatile=True)
            trg = Variable(trg.data.cuda(), volatile=True)
        else:
            src = Variable(src.data, volatile=True)
            trg = Variable(trg.data, volatile=True)
        output = model(src, trg, teacher_forcing_ratio=0.0)
        loss = F.nll_loss(input=output[1:].view(-1, vocab_size),
                          target=trg[1:].contiguous().view(-1),
                          ignore_index=pad)
        # total_loss += loss.data[0]
        total_loss += loss.item()
    return total_loss / len(val_iter)


if __name__ == '__main__':
    # parse_arguments
    batch_size = 32
    lr = 0.0001
    epochs = 100
    grad_clip = 10.0

    hidden_size = 512
    embed_size = 256
    # assert torch.cuda.is_available()
    cuda_available = torch.cuda.is_available()

    # 1. preparing dataset
    print("[!] preparing dataset...")
    train_iter, val_iter, test_iter, DE, EN = load_dataset(batch_size)
    de_size, en_size = len(DE.vocab), len(EN.vocab)
    print("[TRAIN]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)"
          % (len(train_iter), len(train_iter.dataset),
             len(test_iter), len(test_iter.dataset)))
    print("[DE_vocab]:%d [en_vocab]:%d" % (de_size, en_size))

    # 2. Instantiating models
    print("[!] Instantiating models...")
    encoder = Encoder(de_size, embed_size, hidden_size,
                      n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, en_size,
                      n_layers=1, dropout=0.5)
    if cuda_available:
        seq2seq = Seq2Seq(encoder, decoder, cuda_available).cuda()
    else:
        seq2seq = Seq2Seq(encoder, decoder)
    optimizer = optim.Adam(seq2seq.parameters(), lr=lr)
    print(seq2seq)

    # 3. Training
    best_val_loss = None
    for e in range(1, epochs + 1):
        train(e, seq2seq, optimizer, train_iter, en_size, grad_clip, DE, EN)
        val_loss = evaluate(seq2seq, val_iter, en_size, DE, EN)
        print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS" % (e, val_loss, math.exp(val_loss)))
