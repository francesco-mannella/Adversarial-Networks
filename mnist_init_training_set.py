# -*- coding: utf-8 -*-
import os
import numpy as np
from mnist import MNIST


class MNIST_manager(object):
    
    def __init__(self):
        # init with the 'data' dir
        mndata = MNIST('./data')
        # Load data
        mndata.load_training()
        # The number of pixels per side of all images
        self.img_side = 28
        # Each input is a raw vector.
        # The number of units of the network
        # corresponds to the number of input elements
        self.n_mnist_pixels = self.img_side ** 2
        # lengths of datatasets patterns
        self.n_train = len(mndata.train_images)
        # convert data into {-1,1} arrays
        self.images = np.vstack(mndata.train_images[:])/255.0
        self.images = 2*self.images - 1
        # shuffle img indices
        self.image_labels = np.hstack(mndata.train_labels)
        self.idcs = np.arange(self.n_train)

    def get_params(self):
        return self.img_side, self.n_mnist_pixels, self.n_train
    
    def order_imgs(self):
        idcs=np.argsort(self.image_labels)
        return self.images[self.idcs], self.image_labels[self.idcs]


    def shuffle_imgs(self):
        np.random.shuffle(self.idcs)
        return self.images[self.idcs], self.image_labels[self.idcs]
    
    def labelled_imgs(self, label):
        idcs = np.where(self.image_labels == label)
        return self.images[idcs], self.image_labels[idcs]

