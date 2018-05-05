import os

import keras.models as models
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

def getImageForGenerator(dataDir):

    # BGR
    image = cv2.imread()

def getSegmentationForGenerator():

    # BGR
    image = cv2.imread()
    
def imageSegmentationGenerator(path):

    lyftdataSet = LyftDataSet()

class LyftDataSet(object):

    def __init__(self):

        self.SampleDataDir = "Train"
        trainannotDir = "CameraSeg"
        trainDir = "CameraRGB"

        self.train_data_rgb = os.path.join(self.SampleDataDir, trainDir)
        self.trainannot_data = os.path.join(self.SampleDataDir, trainannotDir)        
        
        self.image_files = glob.glob( os.path.join( self.train_data_rgb, "*.png" ) )
        self.seg_files = glob.glob( os.path.join( self.trainannot_data, "*.png" )  )
        
        self.read_label_tag()

    def read_label_tag(self):

        self.df_label = pd.read_csv("label.csv")
        self.n_classes = len(self.df_label)

    def normalized(self,image):

        new_image = np.zeros_like(image)
        r = image[:,:,0]
        g = image[:,:,1]
        b = image[:,:,2]

        new_image[:,:,0] = cv2.equalizeHist(r)
        new_image[:,:,1] = cv2.equalizeHist(g)
        new_image[:,:,2] = cv2.equalizeHist(b)

        return new_image

    def setXy(self):
        image_ops = lambda x:cv2.imread(x)
        segs_ops = lambda x:cv2.imread(x)[:,:,2] # bgr --> r:2 channel
        norm_ops = lambda x:self.normalized(x)

        ims = list( map( norm_ops, list(map( image_ops, self.image_files)   ) ) )
        segs = list( map(  segs_ops, self.seg_files ))

        return np.array(ims), np.array(segs)


