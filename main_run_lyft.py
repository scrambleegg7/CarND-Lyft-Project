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
from SegNet import SegNet
from LyftDataGenerator import LyftDataGenerator
from LyftDataSet import LyftDataSet

def generator(dataSet, batch_size=32, mode="train"):

    while 1: # Loop forever so the generator never terminates
        
        #
        #    shuffle(samples)
        #
        #  
        #dataSet = LyftDataSet(mode=mode)
        if mode == "train":
            num_samples = dataSet.train_num_samples
        else:
            num_samples = dataSet.test_num_samples

        
        for offset in range(0, num_samples, batch_size):

            X, y = dataSet.batch_next(offset, batch_size, mode)

            yield X, y


def build_model():

    input_shape=(600, 800, 3) 
    #classes=13
    #input_shape=(360, 480, 3)

    # Roads, Vehicles and Others
    classes=3


    segNet = SegNet()
    model = segNet.build_model(input_shape, classes)
    model.summary()

    return model

def main():

    model = build_model()


    BATCH_SIZE = 4
    EPOCHS = 1

    dataSet = LyftDataSet()    
    
    train_generator = generator(dataSet,BATCH_SIZE,mode="train")
    validation_generator = generator(dataSet,BATCH_SIZE,mode="test")

    #train_generator = LyftDataGenerator(mode="train", batch_size=16)
    #validation_generator = LyftDataGenerator(mode="test", batch_size=16)
    
    #optimizer = Adam(lr=)
    #model.compile(optimizer=optimizer, loss='mse')
    model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])
    model_file = os.path.join("save_models","lyft_segnet_weights.h5")
    if os.path.isfile(model_file):
        print("** weights file found ....", model_file)

        model.load_weights(model_file)

    steps_per_epch = dataSet.train_num_samples // BATCH_SIZE
    valid_steps = dataSet.test_num_samples // BATCH_SIZE    
    
    history_object = model.fit_generator(train_generator,  steps_per_epoch=steps_per_epch, \
        validation_data=validation_generator, validation_steps= valid_steps,  \
        epochs=EPOCHS, verbose=1)
        #use_multiprocessing=True, workers=4 , epochs=EPOCHS, verbose=1)
        

    model.save_weights(model_file)




if __name__ == "__main__":
    main()