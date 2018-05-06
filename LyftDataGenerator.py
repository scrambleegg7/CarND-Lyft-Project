import numpy as np
import keras
import math


import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.utils import shuffle

# user defined ..
from LyftDataSet import LyftDataSet

#
# special approach to just select Vehicle(10) and Roads(7)
# 
# output layer shoudl be l x h x w x 3
#

class LyftDataGenerator(keras.utils.Sequence):

    'Generates data for Keras'
    def __init__(self, mode="train", batch_size=32, n_classes = 3 ) :
        
        # Initialization
        lytfdataSet = LyftDataSet()
        X, y = lytfdataSet.setXy() 
        X, y = shuffle( X, y )
        x_train , x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15) 

        if mode == "train":
            self.X = x_train
            self.y = y_train
        elif mode == "test":
            self.X = x_test
            self.y = y_test

        print("loaded X y shape")
        print(self.X.shape, self.y.shape)

        l, h, w, c = self.X.shape        
        self.data_length = l
        self.n_channels = c
        
        self.batch_size = batch_size
        self.n_classes = n_classes
        
        print("batch_size", batch_size)
        print("number of labels", self.n_classes)

    def on_epoch_end(self):
        self.X, self.y = shuffle(self.X, self.y)

    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, idx):
        # read data with batch size
        #print("__getitem__ idx : ", idx)

        # idx is assigned with random ?
        #
        batch_x = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.one_hot_labels(batch_y)

        l,h,w,c = batch_y.shape
        batch_y = np.reshape( batch_y,  (l,h * w, c) )

        return batch_x, batch_y

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

def main():

    dg = LyftDataGenerator(batch_size=16)

    X, y = dg.__getitem__(1)
    print(X.shape, y.shape)

    
    
    # test to display segmentation 
    #y = dg.y[0]
    y = y[0].reshape(600,800,3)
    # Roads
    plt.imshow(y[:,:,1])
    plt.show()

    # Vehicles
    plt.imshow(y[:,:,2])
    plt.show()
    

if __name__ == "__main__":
    main()