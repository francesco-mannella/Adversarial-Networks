# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from mlp import MLP, linear
plt.ion()
"""
 Autoencoder

* A multilayered perceptron as the autoencoder network 
    takes samples for the mnist dataset and reproduces them
    
    encoder:  X(28*28) -> H1(512) -> H2(256) -> Z(2)
    decoder:  Z(2)     -> H1(256) -> H2(512) -> Y(28*28)
   
    minimizes:
        D_loss =  mean(sqrt( Y - X)**2))

"""

#-------------------------------------------------------------------------------
# only current needed GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#-------------------------------------------------------------------------------
# set the seed for random numbers generation
import os
current_seed = np.fromstring(os.urandom(4), dtype=np.uint32)[0]
print "seed:%d" % current_seed
rng = np.random.RandomState(current_seed) 
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# mnist
# import the mnist class
from mnist import MNIST
# init with the 'data' dir
mndata = MNIST('./data')
# Load data
mndata.load_training()
# The number of pixels per side of all images
img_side = 28
# Each input is a raw vector.
# The number of units of the network
# corresponds to the number of input elements
n_mnist_pixels = img_side * img_side
# lengths of datatasets patterns
n_train = len(mndata.train_images)
# convert data into {-1,1} arrays
data = np.vstack(mndata.train_images[:])/255.0
data = 2*data - 1
np.random.shuffle(data)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#------------------------------------------------------------------------------- 
# Plot
class Plotter(object):  
    def __init__(self):    
        
        self.fig = plt.figure(figsize=(5,8))
        gs = gridspec.GridSpec(8, 5)

        
        self.losses_ax = self.fig.add_subplot(gs[:3,:])
        self.losses_ax.set_title("Reconstruction error")
        self.losses_lines = []
        line, = self.losses_ax.plot(0,0)   
        self.losses_lines.append(line)
        self.labels = ["reconstruction"]  
        self.losses_ax.legend(self.losses_lines, self.labels)  
        
        # reconstructed patterns          
        self.pattern_axes = []
        self.pattern_imgs = []
        for x in range(5):
            for y in range(5):
                ax = self.fig.add_subplot(gs[3+x, y])
                im = ax.imshow(np.zeros([img_side, img_side]), 
                               vmin=-1, vmax=1)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                self.pattern_axes.append(ax)
                self.pattern_imgs.append(im)

    def plot(self, R_loss, patterns):
        
        losses = [R_loss]   
        t = len(R_loss)
        self.losses_lines[0].set_data(np.arange(t), R_loss)
        self.losses_ax.set_xlim([0,t])  
        self.losses_ax.set_ylim([np.min(losses), np.max(losses)])  
                                    
        l = len(patterns)
        for x in range(5):
            for y in range(5):
                k = x*5 + y     
                im = self.pattern_imgs[k]
                if k<l:
                    im.set_data(patterns[k].reshape(img_side, img_side))
        self.fig.canvas.draw()
        self.fig.savefig("ae.png")
                 
plotter = Plotter()
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#------------------------------------------------------------------------------- 
#Globals 
epochs = 1000
num_samples = 100
eps = 1e-2
lr = 0.0005
#-------------------------------------------------------------------------------
graph = tf.Graph()
with graph.as_default():
    #---------------------------------------------------------------------------
    #  Encoder
    with tf.variable_scope('Encoder'):
        encoder_layers = [n_mnist_pixels, 512, 256, 2]
        encoder_outfuns =[tf.nn.relu, tf.nn.relu, tf.nn.relu] 
        encoder = MLP(lr=lr, drop_out=0.3, scope="E",
                             outfuns=encoder_outfuns, 
                             layers=encoder_layers)   
        data_sample = tf.placeholder(tf.float32, [num_samples, encoder_layers[0]])
        hidden_patterns = encoder.update(data_sample) 
    #---------------------------------------------------------------------------
    #  Decoder
    with tf.variable_scope('decoder'):
        decoder_layers = [2, 512, 256, n_mnist_pixels]
        decoder_outfuns =[tf.nn.relu, tf.nn.relu, tf.nn.tanh] 
        decoder = MLP(lr=lr, drop_out=0.3, scope="E",
                             outfuns=decoder_outfuns, 
                             layers=decoder_layers)   
        
        decoded_patterns = decoder.update(hidden_patterns) 
    #---------------------------------------------------------------------------   
    R_loss = tf.losses.mean_squared_error(data_sample, decoded_patterns)
    #---------------------------------------------------------------------------
    ER_train =  encoder.train(R_loss)
    DR_train =  decoder.train(R_loss)
    #---------------------------------------------------------------------------   
    def get_data_sample(t):
        return np.vstack(data[t*num_samples:(t+1)*num_samples])      
    #---------------------------------------------------------------------------   
    #---------------------------------------------------------------------------   
    #---------------------------------------------------------------------------   
    # Tf Session
    with tf.Session(config=config) as session:
        # writer = tf.summary.FileWriter("output", session.graph)
        session.run(tf.global_variables_initializer())
        R_losses = []
        for epoch in range(epochs):
        
            r_losses = []
            np.random.shuffle(data)
            for t in range(len(data)//num_samples):
            
                # reconstruction step -- encoder -> decoder (minimize reconstruction  error)
                curr_data_sample = get_data_sample(t)
                current_decoded_patterns, r_loss, _, _= session.run(
                    [decoded_patterns, R_loss, ER_train, DR_train], 
                    feed_dict={data_sample:curr_data_sample})     
                
                r_losses.append(r_loss)
            R_losses.append(np.mean(r_losses))
            plotter.plot(R_losses, current_decoded_patterns)
                
