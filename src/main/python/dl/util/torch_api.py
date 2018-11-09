# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/9
version : 
"""
# ------ rnn gru eaxmple
"""
Examples::

>>> rnn = nn.RNN(10, 20, 2)
>>> input = torch.randn(5, 3, 10)
>>> h0 = torch.randn(2, 3, 20)
>>> output, hn = rnn(input, h0)
"""