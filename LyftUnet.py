### Import libraries

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
% matplotlib inline
import glob

from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D,Lambda, Conv2D, MaxPool2D, concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from scipy.ndimage.measurements import label
import time


class LyftUnet(object):

    def buildModel(self, input_width=800 , input_height=600 , nChannels=1 ): 
        
        inputs = Input((input_height, input_width,3))
        inputs_norm = Lambda(lambda x: x/127.5 - 1.)

        # conv1
        conv1 = Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same')(inputs)
        pool1 = MaxPool2D(strides=(2,2))(conv1)
        # conv2
        conv2 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(pool1)
        pool2 = MaxPool2D(strides=(2,2))(conv2)
        # conv3
        conv3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(pool2)
        pool3 = MaxPool2D(strides=(2,2))(conv3)
        # conv4
        conv4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(pool3)
        pool4 = MaxPool2D(strides=(2,2))(conv4)
        #conv5
        conv5 = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(pool4)
        # up6
        up6 = concatenate([UpSampling2D(size=(2,2))(conv5), conv4], axis=-1)
        #conv6
        conv6 = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(up6)
        # up7
        up7 = concatenate([UpSampling2D(size=(2,2))(conv6), conv3], axis=-1)
        #conv7
        conv7 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(up7)
        # up8
        up8 = concatenate([UpSampling2D(size=(2,2))(conv7), conv2], axis=-1)
        #conv8
        conv8 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(up8)
        # up9
        up9 = concatenate([UpSampling2D(size=(2,2))(conv8), conv1], axis=-1)
        # conv9
        conv9 = Conv2D(filters=8, kernel_size=(3,3), activation='relu',  padding='same')(up9)
        #l = Dropout(0.5)(l)
        conv10 = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid')(conv9)

        model = Model(inputs, conv10)

        return model
