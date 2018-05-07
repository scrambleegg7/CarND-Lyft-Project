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

from LyftDataSet import LyftDataSet
from LyftUnet import LyftUnet

from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D,Lambda, Conv2D, MaxPool2D, concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.models import load_model

import keras.losses

### IOU or dice coeff calculation

def IOU_calc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    
    return 2*(intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def IOU_calc_loss(y_true, y_pred):
    return -IOU_calc(y_true, y_pred)


#### Training generator, generates augmented images
def generate_train_batch(data,batch_size = 32):
    
    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_masks = np.zeros((batch_size, img_rows, img_cols, 1))
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(data)-2000)
            name_str,img,bb_boxes = get_image_name(df_vehicles,i_line,
                                                   size=(img_cols, img_rows),
                                                  augmentation=True,
                                                   trans_range=50,
                                                   scale_range=50
                                                  )
            img_mask = get_mask_seg(img,bb_boxes)
            batch_images[i_batch] = img
            batch_masks[i_batch] =img_mask
        yield batch_images, batch_masks
        
#### Testing generator, generates augmented images
def generate_test_batch(data,batch_size = 32):
    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_masks = np.zeros((batch_size, img_rows, img_cols, 1))
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(2000)
            i_line = i_line+len(data)-2000
            name_str,img,bb_boxes = get_image_name(df_vehicles,i_line,
                                                   size=(img_cols, img_rows),
                                                  augmentation=False,
                                                   trans_range=0,
                                                   scale_range=0
                                                  )
            img_mask = get_mask_seg(img,bb_boxes)
            batch_images[i_batch] = img
            batch_masks[i_batch] =img_mask
        yield batch_images, batch_masks


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

    model = LyftUnet().buildModel1()
    model.summary()

    return model

def main():

    model = build_model()


    #x, y = generator_train_batach(dataSet,16)
    #x_test, y_test = generator_train_batach(dataSet,16, mode="test")
    # 
    print("- "* 30)
    print("- confirmed generator offset / batch size working well to get the dataset....")
    offset = 0
    BATCH_SIZE = 4
    EPOCHS = 20

    print("- "* 30)

    dataSet = LyftDataSet()

    train_generator = generator(dataSet,BATCH_SIZE,mode="train")
    validation_generator = generator(dataSet,BATCH_SIZE,mode="test")

    model_file = os.path.join("save_models","unet_weights.h5")
    model.compile(optimizer=Adam(lr=1e-4), 
                loss=IOU_calc_loss, metrics=[IOU_calc])

    if os.path.isfile(model_file):
        print("** weights file found ....", model_file)

    model.load_weights(model_file)

    #model.compile(optimizer=optimizer, loss='mse')
    #model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

    steps_per_epch = dataSet.train_num_samples // BATCH_SIZE
    valid_steps = dataSet.test_num_samples // BATCH_SIZE

    history_object = model.fit_generator(train_generator, steps_per_epoch=steps_per_epch , \
        validation_data=validation_generator,  \
        validation_steps= valid_steps  , epochs=EPOCHS, verbose=1)
    

    model_file = os.path.join("save_models","unet_weights.h5")
    model.save_weights(model_file)



if __name__ == "__main__":
    main()