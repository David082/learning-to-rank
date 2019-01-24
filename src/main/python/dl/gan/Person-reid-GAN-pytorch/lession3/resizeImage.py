# -*- coding: utf-8 -*-
"""
describe :
author : David
created on : 2019/1/14
version :
refer :
"""
import os
from PIL import Image

if __name__ == '__main__':
    fileout = 'resize_market'
    filein = 'gen_market'
    if not os.path.isdir(fileout):
        os.mkdir(fileout)
    for name in os.listdir(filein):
        img = Image.open(filein + '/' + name)
        out = img.resize((64, 128), Image.ANTIALIAS)
        out.save(fileout + '/' + name)
