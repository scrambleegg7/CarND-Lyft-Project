import os

import keras.models as models
import matplotlib.pyplot as plt

from keras import backend as K
from keras.applications import imagenet_utils
from sklearn.utils import shuffle

from skimage.io import imread

import cv2 
import numpy as np
import json
import keras

import glob
import seaborn as sns
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

# user defined module
from SegNet import SegNet
from LyftDataSet import LyftDataSet


def testDataLoad():

    print("** loading data...")

    lytfdataSet = LyftDataSet()
    X, y = lytfdataSet.setXy() 
    X, y = shuffle( X, y )

    n_classes = lytfdataSet.n_classes
    x_train , x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15) 
    
    return x_test[:10], y_test[:10], n_classes


def model_load():

    print("** loading model...")

    model_file = os.path.join("save_models","lyft_full_label_segnet_model_20180506.h5")    

    model = keras.models.load_model(model_file)

    return model

def predict(images, model, n_classes):

    _, h, w, c = images.shape
    probs = model.predict(images, batch_size=1)

    print("** probs shape .", probs.shape)
    #prob = probs[0].reshape((h, w, n_classes)).argmax(axis=2)
    return prob
    
def main():

    x_test, y_test, n_classes = testDataLoad() # top10
    h,w,c = x_test[0].shape

    x_test = x_test[0].reshape( (1,h,w,c) )
    model = model_load()

    prob = predict(x_test,model, n_classes)
    # probability 
    prob = prob[0].reshape(h,w,n_classes).argmax(axis=2)


if __name__ == "__main__":
    main()