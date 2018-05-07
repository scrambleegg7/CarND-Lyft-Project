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


def build_model():

    #input_shape=(600, 800, 3) 
    #classes=13
    input_shape=(360, 480, 3)

    # Roads, Vehicles and Others
    classes=3


    segNet = SegNet()
    model = segNet.build_model(input_shape, classes)
    model.summary()

    return model

def main():

    model = build_model()


    BATCH_SIZE = 16
    EPOCHS = 1
    

    train_generator = LyftDataGenerator(mode="train", batch_size=16)
    validation_generator = LyftDataGenerator(mode="test", batch_size=16)
    
    #optimizer = Adam(lr=)
    #model.compile(optimizer=optimizer, loss='mse')
    model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

    steps_per_epch = 850 // BATCH_SIZE
    valid_steps = 150 // BATCH_SIZE 
    
    history_object = model.fit_generator(train_generator,  steps_per_epoch=steps_per_epch, \
        validation_data=validation_generator, validation_steps= valid_steps,  \
        epochs=EPOCHS, verbose=1)
    
        #use_multiprocessing=True, workers=6 , epochs=EPOCHS, verbose=1)
    

    model_file = os.path.join("save_models","lyft_segnet_model.h5")
    model.save(model_file)



if __name__ == "__main__":
    main()