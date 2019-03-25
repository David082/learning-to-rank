# -*- coding: utf-8 -*-

import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms
import torch
from PIL import ImageFile


class ImageSet(data.Dataset):
    def __init__(self, data_txt, data_transforms):
        f = open(data_txt, "r")
        data_list = []
        label_list = []
        while True:
            line = f.readline()
            if line:
                line = line.strip()
                line_temp = line.split('\t')
                data_path = line_temp[0]
                label = int(line_temp[1])
                data_list.append(data_path)
                label_list.append(label)
            else:
                break
        f.close()
        self.data_list = data_list
        self.label_list = label_list
        self.transforms = data_transforms

    def __getitem__(self, index):
        data_path = self.data_list[index]
        label = self.label_list[index] - 1
        data = Image.open(data_path)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        if data.mode != 'RGB':
            data = data.convert("RGB")
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.data_list)
