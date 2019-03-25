# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/11/23
version :
refer :
https://github.com/philferriere/cocoapi/blob/master/PythonAPI/demos/pycocoDemo.ipynb
"""
from pycocotools.coco import COCO
import os, sys, zipfile
import urllib.request
import shutil
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

if __name__ == '__main__':
    pylab.rcParams['figure.figsize'] = (8.0, 10.0)

    # Record package versions for reproducibility
    print("os: %s" % os.name)
    print("sys: %s" % sys.version)
    print("numpy: %s, %s" % (np.__version__, np.__file__))

    # Setup data paths
    dataDir = os.getcwd() + "/coco"
    dataType = 'val2017'
    annDir = '{}/annotations'.format(dataDir)
    annZipFile = '{}/annotations_train{}.zip'.format(dataDir, dataType)
    annFile = '{}/instances_{}.json'.format(annDir, dataType)
    annURL = 'http://images.cocodataset.org/annotations/annotations_train{}.zip'.format(dataType)
    print(annDir)
    print(annFile)
    print(annZipFile)
    print(annURL)

    # Download data if not available locally
    if not os.path.exists(annDir):
        os.makedirs(annDir)
    if not os.path.exists(annFile):
        if not os.path.exists(annZipFile):
            print("Downloading zipped annotations to " + annZipFile + " ...")
            with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
        print("Unzipping " + annZipFile)
        with zipfile.ZipFile(annZipFile, "r") as zip_ref:
            zip_ref.extractall(dataDir)
        print("... done unzipping")
    print("Will use annotations in " + annFile)
