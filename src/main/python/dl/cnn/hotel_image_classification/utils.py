# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/15
version : 
"""
import torch
import pandas as pd


def change_image_link(txt_path):
    df = pd.read_csv(txt_path, sep="\t", header=None)
    df.columns = ["link", "label"]
    df["link"] = df["link"].map(
        lambda x: x.replace("lixianga/hotel_image_classification_code_demo", "yuwei/hotel_image_classification/data"))
    df.to_csv(txt_path, sep="\t", index=False, header=False)


if __name__ == '__main__':
    change_image_link("data/train.txt")
    change_image_link("data/test.txt")

    # https://stackoverflow.com/questions/41861354/loading-torch7-trained-models-t7-in-pytorch
    from torch.utils.serialization import load_lua
    model = load_lua("resnet50_places365.t7", unknown_classes=True, long_size=8)
    # https://github.com/bobbens/sketch_simplification/issues/2
    model = load_lua("resnet50_places365.t7", unknown_classes=True, long_size=8)
