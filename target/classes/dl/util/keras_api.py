# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/23
version : 
"""
# ------ keras卷积补零相关的border_mode的选择以及padding的操作
# https://blog.csdn.net/lujiandong1/article/details/54918320
"""
keras卷积操作中border_mode的实现
如果卷积的方式选择为same,那么卷积操作的输入和输出尺寸会保持一致。如果选择valid,那卷积过后,尺寸会变小。

If tuple of int (length 2): How many zeros to add at the beginning and end of the 2 padding dimensions (rows and cols)
说明：这是keras中的补零操作,下面举2个例子。
padding= (1,0),会在行的最前和最后都增加一行0。比方说,原来的尺寸为(None,20,11,1),padding之后就会变成(None,22,11,1).
padding= (1,1),会在行和列的最前和最后都增加一行0。比方说,原来的尺寸为(None,20,11,1),padding之后就会变成(None,22,13,1).
说明：在复现某个网络的过程中,作者使用的是宽卷积,也即长度为N,滤波器的大小为w,那么卷积之后的结果为N+W-1。这个跟数字信号处理中的卷积操作是一致的。这个就没办法用keras的API,所以直接使用padding就可以了。
"""