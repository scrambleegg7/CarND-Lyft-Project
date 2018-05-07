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

        print("images / labels filename is loading ...")
        self.train_data_rgb = os.path.join(self.SampleDataDir, trainDir)
        self.trainannot_data = os.path.join(self.SampleDataDir, trainannotDir)        
        
        self.image_files = glob.glob( os.path.join( self.train_data_rgb, "*.png" ) )
        self.seg_files = glob.glob( os.path.join( self.trainannot_data, "*.png" )  )

        self.Xtrain_files , self.Xtest_files, self.ytrain_files, self.ytest_files = train_test_split(self.image_files, self.seg_files , test_size = 0.15)         
        print("loading done ... train test sizes : ", len(self.Xtrain_files), len(self.Xtest_files))

        self.train_num_samples = len(self.Xtrain_files)
        self.test_num_samples = len(self.Xtest_files)

        self.n_classes = 3

        #self.read_label_tag()
    
    def getFilenames(self):
        return self.image_files, self.seg_files
    
    #
    # batch next is called with generator
    #
    def batch_next(self,offset,BATCH_SIZE=32, mode="train", unet=False):

        # get filenames by BATCH SIZE
        if mode == "train":
            image_samples = self.Xtrain_files[ offset:offset+BATCH_SIZE ]        
            label_samples = self.ytrain_files[ offset:offset+BATCH_SIZE ]
        elif mode == "test":
            image_samples = self.Xtest_files[ offset:offset+BATCH_SIZE ]        
            label_samples = self.ytest_files[ offset:offset+BATCH_SIZE ]
            
        #print(len( image_samples ), len( label_samples ))    
        images = self.preprocess_image(image_samples)
        labels = self.preprocess_label(label_samples,unet)

        #print(images.shape)
        #print(labels.shape)

        return shuffle( images, labels )    

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
    
    def subMean(self,image):
        
        img = image.copy().astype(np.float32)
        img[:,:,0] -= 103.939
        img[:,:,1] -= 116.779
        img[:,:,2] -= 123.68
        
        return img
    
    def preprocess_image(self,image_files):

        image_ops = lambda x: cv2.imread(x)
        norm_ops = lambda x:self.normalized(x)
        subMean_ops = lambda x:self.subMean(x)
        ims = list( map( norm_ops, list(map( image_ops, image_files)   ) ) )
        ims = list( map( subMean_ops, ims ) )

        return np.array(ims)

    def preprocess_label(self, seg_files, unet=False):

        # file read with original image size
        segs_ops = lambda x:  cv2.imread(x)[:,:,2] # bgr --> r:2 channel
        segs = list( map(  segs_ops, seg_files ))

        # for Vehicles 

        if unet:
            segs = self.one_hot_labels_unet( np.array( segs ),label=10)
            # change shape to fit keras output model ( None, h, w, c)
            l,h,w = segs.shape
            segs = np.reshape( segs, (l,h,w,1))
        else:
            segs = self.one_hot_labels( np.array( segs ))
            l,h,w,c = segs.shape
            segs = np.reshape( segs, (l,h * w, c))
            

        return np.array( segs )

    def one_hot_labels_unet(self, labels, label = 7):

        l,h,w = labels.shape
        new_labels = np.zeros_like(labels)
        
        for i in range(l):
            bins = np.zeros_like( labels[i] )
            bins[ labels[i] == label ] = 1
            new_labels[i] = bins

            if label == 10:
                label_front = new_labels[i,:490,:].copy()
                label_hood = new_labels[i,490:,:].copy()
                label_hood[label_hood == 1] = 0
                new_labels[i,:,:] = np.vstack( [label_front, label_hood] )



        return new_labels

    def one_hot_labels(self, labels):

        l,h,w = labels.shape
        new_labels = np.zeros([l,h,w,self.n_classes])
        
        # for debug new_labels shape
        #print(new_labels.shape)

        # new_labels[:,:,0] None
        # new_labels[:,:,1] Roads
        # new_labels[:,:,2] Vehicles 

        for i in range(l):
            for idx, j in enumerate( [0,7,10] ):
                
                bins = np.zeros_like( labels[i])
                bins[ labels[i] == j ] = 1

                new_labels[i,:,:,idx] = bins

            label_front = new_labels[i,:490,:,2].copy()
            label_hood = new_labels[i,490:,:,2].copy()
            label_hood[label_hood == 1] = 0
            new_labels[i,:,:,2] = np.vstack( [label_front, label_hood] )
        
        return new_labels


    def setXy(self):

        # img = cv2.resize(img, ( width , height ))
        image_ops = lambda x: cv2.resize( cv2.imread(x), (480, 360) )  

        segs_ops = lambda x:  cv2.imread(x)[:,:,2] # bgr --> r:2 channel
        
        norm_ops = lambda x:self.normalized(x)
        subMean_ops = lambda x:self.subMean(x)

        ims = list( map( norm_ops, list(map( image_ops, self.image_files)   ) ) )
        ims = list( map( subMean_ops, ims ) )

        segs = list( map(  segs_ops, self.seg_files ))

        return np.array(ims), np.array(segs)




def main():

    lyftDataSet = LyftDataSet()

    X, y = lyftDataSet.batch_next(0,BATCH_SIZE=16, mode="test")

    plt.imshow(y[10])
    plt.show()


    lyftDataSet = LyftDataSet()

    X, y = lyftDataSet.batch_next(10,BATCH_SIZE=16,mode="train")

    plt.imshow(y[10])
    plt.show()

if __name__ == "__main__":
    main()