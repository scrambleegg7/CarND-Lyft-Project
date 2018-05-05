import os

import keras.models as models
#from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
#from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
#from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt

from keras import backend as K
from keras.applications import imagenet_utils

from skimage.io import imread
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.utils import shuffle

import cv2
import numpy as np
import json

import glob
import seaborn as sns
import pandas as pd

class DataSet(object):

    def __init__(self):

        self.SampleDataDir = "CamVid"
        self.trainannotDir = "trainannot"
        self.trainDir = "train"
        train_txt = "train.txt"
        test_txt = "test.txt"

        train_data = os.path.join(self.SampleDataDir, train_txt)
        test_data = os.path.join(self.SampleDataDir, test_txt)

        # create dataframe object to read train.txt
        self.df_train = pd.read_csv(train_data,sep="\s+", names=["image","label"])
        self.df_test = pd.read_csv(test_data,sep="\s+", names=["image","label"])

    def normalized(self, rgb):
        #return rgb/255.0
        norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

        b=rgb[:,:,0]
        g=rgb[:,:,1]
        r=rgb[:,:,2]

        norm[:,:,0]=cv2.equalizeHist(b)
        norm[:,:,1]=cv2.equalizeHist(g)
        norm[:,:,2]=cv2.equalizeHist(r)

        return norm

    def load_data(self, mode = "train"):

        ## lambda operation 
        split_ops = lambda x:x.split("/")[-1]

        if mode == "train":
            image_files = self.df_train["image"].apply(split_ops).tolist()
            label_files = self.df_train["label"].apply(split_ops).tolist()
        else:
            image_files = self.df_test["image"].apply(split_ops).tolist()
            label_files = self.df_test["label"].apply(split_ops).tolist()

        image_ops = lambda x:imread(x)
        traindir_ops = lambda x:os.path.join(self.SampleDataDir+"/"+ mode,   x)
        trainannotdir_ops = lambda x:os.path.join(self.SampleDataDir+"/"+ mode + "annot",x)

        print("-" * 30)
        print(" mode : %s " % mode )
        print(" loading image and label from data dir ")
        print(" data dir --> %s " % self.SampleDataDir+"/"+ mode )
        print(" annotation  dir --> %s " % self.SampleDataDir+"/"+ mode +  "annot")
        
        images =    list(  map( image_ops,    list(map(  traindir_ops, image_files)) ) )
        labels =    list(  map( image_ops,    list(map(  trainannotdir_ops, label_files)) ) )

        images = np.array(images)
        labels = np.array(labels)

        return images, labels
    
    def one_hot_labels(self, labels):

        print("-" * 30)
        print("* one_hot_labels *")
        print(" convert single label to one hot labels --> m x n x 12 ")
        l,h,w = labels.shape
        new_labels = np.zeros([l,h,w,12])
        print(new_labels.shape)

        for i in range(l):
            for j in range(12):
                
                bins = np.zeros_like( labels[i])
                bins[ labels[i] == j ] = 1
                new_labels[i,:,:,j] = bins
        
        return new_labels

    def makeTrainData(self, mode="train"):
        images, labels = self.load_data(mode)

        # normalized 
        normalized_ops = lambda x:self.normalized(x)
        norm_images = list(  map(  normalized_ops,  images    )  )
        norm_images = np.array(norm_images)

        labels = self.one_hot_labels(labels)

        return norm_images, labels
    
    def caffeStyleConvert(self,images):
        x = imagenet_utils.preprocess_input(images)

        return x
    
    def setXY(self):

        images, labels = self.makeTrainData()
        l,h,w,c = labels.shape
        # keras preprocess_input 
        # for subtracting caffe mean
        # based on VGG16
        self.X = self.caffeStyleConvert(images)
        self.y = np.reshape( labels, (l,h*w,c)  )

        images, labels = self.makeTrainData("test")
        l,h,w,c = labels.shape
        # keras preprocess_input 
        # for subtracting caffe mean
        # based on VGG16
        self.X_test = self.caffeStyleConvert(images)
        self.y_test = np.reshape( labels, (l,h*w,c)  )


    def batch_next(self,offset,BATCH_SIZE=32, mode="train"):

        if mode == "train":
            image_samples = self.X[ offset:offset+BATCH_SIZE ]        
            label_samples = self.y[ offset:offset+BATCH_SIZE ]
        else:
            image_samples = self.X_test[ offset:offset+BATCH_SIZE ]        
            label_samples = self.y_test[ offset:offset+BATCH_SIZE ]
            

        return shuffle( image_samples, label_samples )    


