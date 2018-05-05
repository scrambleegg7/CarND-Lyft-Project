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

import glob
import seaborn as sns
import pandas as pd

# user defined module
from DataSet import DataSet
from SegNet import SegNet

dataSet = DataSet()


def generator(dataSet, batch_size=32, mode="train"):

    while 1: # Loop forever so the generator never terminates
        
        #
        #    shuffle(samples)
        #
        #  
        X = dataSet.X
        num_samples, h, w, c = X.shape

        for offset in range(0, num_samples, batch_size):

            X, y = dataSet.batch_next(offset,batch_size, mode)

            yield X, y

def load_data():

    dataSet.setXY()

def build_model():

    segNet = SegNet()
    model = segNet.build_model()
    model.summary()

    return model

def main():

    model = build_model()

    load_data()
    X = dataSet.X
    X_ = dataSet.X_test

    #x, y = generator_train_batach(dataSet,16)
    #x_test, y_test = generator_train_batach(dataSet,16, mode="test")
    # 
    print("- "* 30)
    print("- confirmed generator offset / batch size working well to get the dataset....")
    offset = 0
    BATCH_SIZE = 16
    EPOCHS = 1
    X_train, y_train = dataSet.batch_next(offset,BATCH_SIZE, "train")
    X_test, y_test = dataSet.batch_next(offset,BATCH_SIZE, "test")
    
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    print("- "* 30)

    train_generator = generator(dataSet,BATCH_SIZE)
    validation_generator = generator(dataSet,BATCH_SIZE,mode="test")
    
    #optimizer = Adam(lr=)
    #model.compile(optimizer=optimizer, loss='mse')
    model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

    steps_per_epch = X.shape[0] // BATCH_SIZE
    valid_steps = X_.shape[0] // BATCH_SIZE 
    history_object = model.fit_generator(train_generator, steps_per_epoch=steps_per_epch , \
        validation_data=validation_generator,  \
        validation_steps= valid_steps  , epochs=EPOCHS, verbose=1)
    

    model_file = os.path.join("save_models","segnet_model.h5")
    model.save(model_file)



if __name__ == "__main__":
    main()