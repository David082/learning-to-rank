# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/12/4
version :
refer :
https://github.com/AllenCX/IMDB-RNN-Attention/blob/master/config.py
"""


class Config(object):
    def __init__(self,
                 batch_size,
                 embedding_size,
                 encoder_hidden_size,
                 vocab_size,
                 lr,
                 epoch_num,
                 save_per_epoch,
                 max_length,
                 max_grad_norm,
                 keep_prob,
                 atn_hidden_size,
                 ckpt_path):
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.encoder_hidden_size = encoder_hidden_size
        self.vocab_size = vocab_size
        self.epoch_num = epoch_num
        self.lr = lr
        self.summary_dir = "summary"
        self.save_per_epoch = save_per_epoch
        self.max_length = max_length
        self.max_grad_norm = max_grad_norm
        self.ckpt_path = ckpt_path
        self.keep_prob = keep_prob
        self.atn_hidden_size = atn_hidden_size
