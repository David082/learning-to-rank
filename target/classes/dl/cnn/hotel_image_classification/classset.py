# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/12/12
version : 
"""
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms
import torch
from PIL import ImageFile


class Images(data.Dataset):
    def __init__(self, imgs_path, img_cls, transforms):
        self.img_cls = img_cls  # image label
        img_list = os.listdir(imgs_path + '/' + str(img_cls))
        # img_list.sort() # un shuffle
        self.img_list = [imgs_path + '/' + str(img_cls) + '/' + i for i in img_list]
        self.transforms = transforms

    def __getitem__(self, index):
        image_path = self.img_list[index]
        # image_label = self.label
        image = Image.open(image_path)
        # ImageFile.LOAD_TRUNCATED_IMAGES = True
        if image.mode != 'RGB':
            image = image.convert("RGB")
        image = self.transforms(image)
        return image, self.img_cls

    def __len__(self):
        return len(self.img_list)


def label_accuracy(model_path, label, image_path=os.getcwd() + '/data/train_images'):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # Tensor image of size (C, H, W) to be normalized.
    test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(), normalize])
    test_set = Images(image_path, label, test_transforms)
    test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=test_set.__len__(), shuffle=False)

    # load model
    model = torch.load(model_path)
    if torch.cuda.is_available():
        model = model.cuda()

    """Sets the module in evaluation mode.
    This has any effect only on certain modules.
    See documentations of particular modules for details of their behaviors in training/evaluation mode, if they are affected, e.g.
    Dropout, BatchNorm, etc.
    """
    model.eval()  # eval

    total = 0
    correct = 0
    acc = 0
    # predict
    for img, label in test_loader:
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()

        total += label.size(0)

        output = model(img)
        # print(output, label)
        _, pred = torch.max(output, 1)
        # print(pred == label)
        correct += (pred == (label - 1)).sum()
        # print(total, correct)

        if total % 100 == 0:
            print('Test Accuracy of the model: %f%%' % (100.0 * correct / total))

        acc = correct / total

    return acc


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    IMAGE_PATH = ""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # Tensor image of size (C, H, W) to be normalized.
    test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(), normalize])
    test_set = Images(IMAGE_PATH, 1, test_transforms)
    test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=test_set.__len__(), shuffle=False)

    # load model
    MODEL_PATH = ""
    model = torch.load(MODEL_PATH + "/net_10.pth")
    if torch.cuda.is_available():
        model = model.cuda()

    """Sets the module in evaluation mode.
    This has any effect only on certain modules.
    See documentations of particular modules for details of their behaviors in training/evaluation mode, if they are affected, e.g.
    Dropout, BatchNorm, etc.
    """
    model.eval()  # eval

    total = 0
    correct = 0

    # predict
    for img, label in test_loader:
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()

        total += label.size(0)

        output = model(img)
        print(output, label)
        _, pred = torch.max(output, 1)
        print(pred == label)
        correct += (pred == (label - 1)).sum()
        # print(total, correct)

        if total % 100 == 0:
            print('Test Accuracy of the model: %f%%' % (100.0 * correct / total))
