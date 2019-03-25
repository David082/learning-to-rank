# -*- coding: utf-8 -*-
# change name of the folder(e.g.  0002,0007,0010,0011...  to 0,1,2,3)
import os
from shutil import copyfile
import argparse

# Options
parser = argparse.ArgumentParser(description='pytorch data path')
parser.add_argument('--data_path', default='pytorch', type=str, help='the generated folder of pytorch data set')
opt = parser.parse_args()


# copy folder tree from source to destination
def copyfolder(src, dst):
    files = os.listdir(src)
    if not os.path.isdir(dst):
        os.mkdir(dst)
    for tt in files:
        copyfile(src + '/' + tt, dst + '/' + tt)


if __name__ == '__main__':
    # original_path = '/home/gq123/guanqiao/deeplearning/reid/market/pytorch'
    original_path = opt.data_path

    data_dir = ['/train', '/val']

    for i in data_dir:
        train_save_path = original_path + i + '_new'
        data_path = original_path + i
        if not os.path.isdir(train_save_path):
            os.mkdir(train_save_path)

        reid_index = 0
        folders = os.listdir(data_path)
        for foldernames in folders:
            copyfolder(data_path + '/' + foldernames, train_save_path + '/' + str(reid_index).zfill(4))
            reid_index = reid_index + 1
