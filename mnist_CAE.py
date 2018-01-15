# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from convolutional_mlp import MLP, linear
plt.ion()
"""
 Convolutional Autoencoder

* A multilayered perceptron  with 2 convolutional layers, 4 all-to-all layers 
    and 2 deconvolutional layers is the autoencoder network. 
    Takes samples for the mnist dataset and reproduces them
    
    encoder:  X(28*28) -> H1(14*14*32) -> H2(14*14*16) -> Z(10)
    decoder:  Z(10)     -> H2(14*14*16) -> H2(14*14*32) -> Y(28*28)
   
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
current_seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]
print "seed:%d" % current_seed
rng = np.random.RandomState(current_seed) 
tf.set_random_seed(current_seed)
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
    def __init__(self, stime=100):    
        
        self.t = 0
        self.fig = plt.figure(figsize=(10,8))
        gs = gridspec.GridSpec(8, 10)

        
        self.losses_ax = self.fig.add_subplot(gs[:2,:])
        self.losses_ax.set_title("Reconstruction error")
        self.losses_lines = []
        line, = self.losses_ax.plot(0,0)   
        self.losses_lines.append(line)
        self.labels = ["reconstruction"]  
        self.losses_ax.legend(self.losses_lines, self.labels)  
        self.losses_ax.set_xlim([0, stime])
        self.losses_ax.set_ylim([0, 0.5])
       
        # data data_patterns          
        self.data_pattern_axes = []
        self.data_pattern_imgs = []
        for x in range(5):
            for y in range(5):
                ax = self.fig.add_subplot(gs[3+x, y])
                im = ax.imshow(np.zeros([img_side, img_side]), 
                               vmin=-1, vmax=1)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                self.data_pattern_axes.append(ax)
                self.data_pattern_imgs.append(im)

        # reconstructed patterns          
        self.pattern_axes = []
        self.pattern_imgs = []
        for x in range(5):
            for y in range(5):
                ax = self.fig.add_subplot(gs[3+x,5 + y])
                im = ax.imshow(np.zeros([img_side, img_side]), 
                               vmin=-1, vmax=1)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                self.pattern_axes.append(ax)
                self.pattern_imgs.append(im)

    def plot(self, R_loss, data, patterns):
        
        losses = [R_loss]   
        t = len(R_loss)
        self.losses_lines[0].set_data(np.arange(t), R_loss)
                                    
        l = len(patterns)
        for x in range(5):
            for y in range(5):
                k = x*5 + y     
                im = self.data_pattern_imgs[k]
                if k<l:
                    im.set_data(data[k].reshape(img_side, img_side))
        for x in range(5):
            for y in range(5):
                k = x*5 + y     
                im = self.pattern_imgs[k]
                if k<l:
                    im.set_data(patterns[k].reshape(img_side, img_side))
        self.fig.canvas.draw()
        self.fig.savefig("ae.png".format(self.t))
        self.fig.savefig("ae-{:03d}.png".format(self.t))
        self.t += 1
                 
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#------------------------------------------------------------------------------- 
#Globals 
epochs = 50
num_samples = 300
lr = 0.3
#-------------------------------------------------------------------------------

plotter = Plotter(epochs)
#-------------------------------------------------------------------------------
graph = tf.Graph()
with graph.as_default():
    #---------------------------------------------------------------------------
    drop_out = tf.placeholder(tf.float32, ())
    #  Autoencoder
    with tf.variable_scope('Autoencoder'):

        # network descripion
        
        #input
        autoencoder_layers    = [ [28, 28, 1]  ]

        # encoder
        autoencoder_layers    += [ [14, 14, 32],   [7, 7, 16],      [100],         [10],          ]
        autoencoder_outfuns    = [ tf.nn.relu,     tf.nn.relu,      tf.nn.relu,    tf.nn.relu     ] 
        autoencoder_convs      = [ [7, 7, 1, 32],  [3, 3, 32, 16],  None,          None           ]
        autoencoder_deconvs    = [ None,           None,            None,          None           ]
        autoencoder_copy_from  = [ None,           None,            None,          None           ]
        autoencoder_strides    = [ 2,              2,               None,          None           ]
        
        # decoder
        autoencoder_layers    += [ [100],          [7, 7, 16],     [14, 14, 32],   [28, 28, 1]    ]
        autoencoder_outfuns   += [ tf.nn.relu,     tf.nn.relu,     tf.nn.relu,     tf.tanh        ] 
        autoencoder_convs     += [ None,           None,           None,           None           ]
        autoencoder_deconvs   += [ None,           None,           [3, 3, 32, 16], [7, 7, 1, 32]  ]
        autoencoder_copy_from += [ 3,              2,              1,              0              ]
        autoencoder_strides   += [ None,           None,           2,              2              ]

        # the autoencoder object 
        autoencoder = MLP(lr=lr, drop_out=0.3, scope="A",
                convs=autoencoder_convs,
                deconvs=autoencoder_deconvs,
                strides=autoencoder_strides,
                outfuns=autoencoder_outfuns, 
                copy_from=autoencoder_copy_from, 
                layers_lens=autoencoder_layers)   
        
        # spreading graph
        data_sample = tf.placeholder(tf.float32, [num_samples] + autoencoder_layers[0])
        reconstructed_sample = autoencoder.update(data_sample, drop_out)      
    #---------------------------------------------------------------------------   
    # loss function
    R_loss = tf.reduce_mean(tf.pow(data_sample - reconstructed_sample, 2.0))
    #---------------------------------------------------------------------------
    # optimization step
    R_train =  autoencoder.train(R_loss, optimizer=tf.train.GradientDescentOptimizer)
    #---------------------------------------------------------------------------   
    def get_data_sample(t):
        sample = np.vstack(data[t*num_samples:(t+1)*num_samples])  
        return sample.reshape(num_samples, img_side, img_side, 1)    
    #---------------------------------------------------------------------------   
    #---------------------------------------------------------------------------   
    #---------------------------------------------------------------------------   
    # Tf Session
    with tf.Session(config=config) as session:
        # writer = tf.summary.FileWriter("output", session.graph)
        session.run(tf.global_variables_initializer())
        R_losses = []
        # iterate over epochs 
        for epoch in range(epochs):
            r_losses = []
            np.random.shuffle(data)
            # iterate ovet batches
            for t in range(len(data)//num_samples):
            
                # reconstruction step -- encoder -> decoder (minimize reconstruction  error)
                curr_data_sample = get_data_sample(t)
                current_decoded_patterns, r_loss, _ = session.run(
                    [reconstructed_sample, R_loss, R_train], 
                    feed_dict={data_sample:curr_data_sample, drop_out: 1.0})       
                r_losses.append(r_loss) 
                
            R_losses.append(np.mean(r_losses))
            
            # test
            np.random.shuffle(data)
            test_data_sample = get_data_sample(0)
            test_decoded_patterns = session.run( 
                reconstructed_sample,
                feed_dict={data_sample:test_data_sample, drop_out: 1.0}) 
            
            # plot
            plotter.plot(R_losses, test_data_sample, test_decoded_patterns)
                
