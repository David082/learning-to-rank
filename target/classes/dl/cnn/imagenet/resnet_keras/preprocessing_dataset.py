# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/22
version : 
"""
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
import shutil


def remove_and_create_class(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)
    os.mkdir(dirname + '/cat')
    os.mkdir(dirname + '/dog')


if __name__ == '__main__':
    train_path = "C:/ImageNet/train"
    test_path = "C:/ImageNet/test"

    # Visualize the size of the original train dataset.
    train_filenames = os.listdir(train_path)
    train_cat = [x for x in train_filenames if x[:3] == 'cat']
    train_dog = [x for x in train_filenames if x[:3] == 'dog']
    x = ['train_cat', 'train_dog', 'test']
    y = [len(train_cat), len(train_dog), len(os.listdir(test_path))]
    ax = sns.barplot(x=x, y=y)

    # Shuffle and split the train filenames
    mytrain, myvalid = train_test_split(train_filenames, test_size=0.1)
    print(len(mytrain), len(myvalid))

    # Visualize the size of the processed train dataset
    mytrain_cat = [x for x in mytrain if x[:3] == 'cat']
    mytrain_dog = [x for x in mytrain if x[:3] == 'dog']
    myvalid_cat = [x for x in myvalid if x[:3] == 'cat']
    myvalid_dog = [x for x in myvalid if x[:3] == 'dog']
    x = ['mytrain_cat', 'mytrain_dog', 'myvalid_cat', 'myvalid_dog']
    y = [len(mytrain_cat), len(mytrain_dog), len(myvalid_cat), len(myvalid_dog)]
    ax = sns.barplot(x=x, y=y)

    # Create symbolic link of images
    remove_and_create_class('C:/ImageNet/mytrain')
    remove_and_create_class('C:/ImageNet/myvalid')

    for filename in mytrain_cat:
        os.symlink('C:/ImageNet/train/' + filename, 'C:/ImageNet/mytrain/cat/' + filename)

    for filename in mytrain_dog:
        os.symlink('C:/ImageNet/train/' + filename, 'C:/ImageNet/mytrain/dog/' + filename)

    for filename in myvalid_cat:
        os.symlink('C:/ImageNet/train/' + filename, 'C:/ImageNet/myvalid/cat/' + filename)

    for filename in myvalid_dog:
        os.symlink('C:/ImageNet/train/' + filename, 'C:/ImageNet/myvalid/dog/' + filename)
