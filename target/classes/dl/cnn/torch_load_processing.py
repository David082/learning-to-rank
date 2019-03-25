# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/9
version :
refer :
PyTorch:数据加载和预处理
https://ranmaosong.github.io/2018/01/05/PyTorch-Data-Loading-and-Processing-TUtorial/
"""
from __future__ import print_function, division
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker=".", c="r")


if __name__ == '__main__':
    path = os.getcwd() + "/resources/cnn/faces/"
    landmarks_frame = pd.read_csv(os.path.join(path, "face_landmarks.csv"))
    n = 65
    img_name = landmarks_frame.ix[n, 0]
    landmarks = landmarks_frame.ix[n, 1:].as_matrix().astype('float')
    landmarks = landmarks.reshape(-1, 2)

    plt.figure()
    img = io.imread(os.path.join("./data/faces/", img_name))
    show_landmarks(io.imread(os.path.join(".data/faces/", img_name)), landmarks)
    plt.show()
    path = os.path.join("./resources/cnn/faces/")
