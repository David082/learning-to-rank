# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/23
version :
refer :
https://github.com/SrikanthVelpuri/Mask_RCNN/blob/master/demo.ipynb

note :
-- ModuleNotFoundError: No module named 'XXX'
https://blog.csdn.net/jianglianye21/article/details/78086768
# sys.path.append

-- AttributeError: module 'keras.engine.topology' has no attribute'load_weights_from_hdf5_group_by_name
keras：pip install keras==2.0.8
"""
import os
import sys

sys.path.append("E:\learning-to-rank\src\main\python\dl\cnn\object_detection\mask_rcnn\mask_rcnn_webcam")

import random
import math
import numpy as np
import skimage.io
import cv2
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize


class InferenceConfig(coco.CocoConfig):
    """Configurations

    We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the CocoConfig class in coco.py.

    For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the CocoConfig class and override the attributes you need to change.
    """
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


if __name__ == '__main__':

    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Directory of images to run detection on
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")

    # ------ 1. Configurations
    config = InferenceConfig()
    config.display()

    # ------ 2. Create Model and Load Trained Weights
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # ------ 3. Class Names
    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    # ------ 4. Run Object Detection
    # Load a random image from the images folder
    file_names = next(os.walk(IMAGE_DIR))[2]
    image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    cv2.imshow("Image", image)

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])
