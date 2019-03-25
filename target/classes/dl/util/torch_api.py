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
# ------ DataLoader
"""
https://www.e-learn.cn/content/qita/850153
https://blog.csdn.net/g11d111/article/details/81504637
"""
# ------ pytorch学习：动量法momentum
"""
https://blog.csdn.net/xckkcxxck/article/details/82319278
"""
# ------ optimizer.zero_grad()
"""
https://blog.csdn.net/scut_salmon/article/details/82414730

# -- zero the parameter gradients
optimizer.zero_grad()
# -- forward + backward + optimize
outputs = net(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
"""