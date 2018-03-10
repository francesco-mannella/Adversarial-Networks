# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from convolutional_mlp import linear, tanh, leaky_relu
from CAE import CAE
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# only current needed GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#-------------------------------------------------------------------------------
# set the seed for random numbers generation
current_seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]
print "seed:%d" % current_seed
rng = np.random.RandomState(current_seed) 
tf.set_random_seed(current_seed)
#-------------------------------------------------------------------------------
# mnist
from mnist_init_training_set import MNIST_manager
mmanager = MNIST_manager()
img_side, n_mnist_pixels, n_train = mmanager.get_params() 
data, data_labels = mmanager.shuffle_imgs()
#-------------------------------------------------------------------------------
# plot
import matplotlib.pyplot as plt
plt.ion()
from mnist_CAE_plot import Plotter
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
"""
Adversarial convolutional autoencoder

* A convolutional network as the autoencoder network 
    takes samples for the mnist dataset and reproduces them
    
    encoder:  X(28,28,1) -> H1(14, 14, 3) -> H2(7, 7, 25) ->  H3(7, 7, 10) -> Z(2)
    decoder:  Z(2)       -> H1(7, 7, 10)  -> H2(7, 7, 25) -> H3(14, 14, 3) -> Y(28,28,1)

    
* A convolutional network as the discriminator network - p = D(Z)
    takes a sample form the hidden pattern space Z and gives a probability 
    that the image belongs to a prior distribution Z_g
    and not the encoder inner activities Z_d
    
    discriminator:  X(1000) -> H1(1000) -> H2(1000) -> Y(1)

    1) the autoencoder minimizes:
        R_loss =  mean(sqrt( Y - X)**2))
    2) the discriminator maximizes:
        D_loss = log(D(Z_d)) + log(1 - D(Z_p) 
    the encoder part of the autoencoder maximizes:
        G_loss = log( D(Z_p) 
"""
#---------------------------------------------------------------------------  
#---------------------------------------------------------------------------  
#---------------------------------------------------------------------------  

# Globals 
epochs = 100
num_samples = 600
eps = 1e-10
rlr = 0.05
tests = 60000
dropout_train = 0.5
dropout_test = 1.0
weight_scale = 0.01
decay = 0.9

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def get_data_sample(t):
    """
    get a batch sample of mnist patterns 
    :param t: the binibatch step within an epoch 
    """
    sample_rng = range(t*num_samples,(t+1)*num_samples)
    return data[sample_rng], data_labels[sample_rng]     

plotter = Plotter(epochs)
#-------------------------------------------------------------------------------

# data used for the test phase
test_data = data[:tests].reshape(tests, img_side, img_side, 1)
test_labels = data_labels[:tests]

#-------------------------------------------------------------------------------
print "globals and samples initialized"
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

graph = tf.Graph()
with graph.as_default():
    
    autoencoder_layers      = [[28, 28, 1]]
    autoencoder_layers     += [[14, 14, 30],  [7, 7, 15],       [7, 7, 5],         [200],              [2]]
    autoencoder_convs       = [[3, 3, 1, 30], [3, 3, 30, 15],   [7, 7, 15, 5],     None,               None]
    autoencoder_deconvs     = [None,          None,             None,              None,               None]
    autoencoder_copy_from   = [None,          None,             None,              None,               None]
    autoencoder_strides     = [2,             2,                1,                 None,               None]
    autoencoder_outfuns     = [tf.nn.relu,    tf.nn.relu,       tf.nn.relu,        tf.nn.relu,         tf.nn.relu]
    autoencoder_bn          = [False,         True,             True,              True,               True] 
    autoencoder_dropouts    = [False,         False,            False,             False,              False] 

    autoencoder_layers     += [[200],         [7, 7, 5],        [7, 7, 15],        [14, 14, 30],       [28, 28, 1]]
    autoencoder_convs      += [None,          None,             None,              None,               None]
    autoencoder_deconvs    += [None,          None,             [7, 7, 15, 5],     [3, 3, 30, 15],     [3, 3, 1, 30]]
    autoencoder_copy_from  += [4,             3,                2,                 1,                  0]
    autoencoder_strides    += [None,          None,             1,                 2,                  2]
    autoencoder_outfuns    += [tf.nn.relu,    tf.nn.relu,       tf.nn.relu,        tf.nn.relu,         linear] 
    autoencoder_bn         += [True,          True,             True,              True,               False] 
    autoencoder_dropouts   += [False,         False,            False,             False,              False]  

    cae = CAE(rlr,  weight_scale, autoencoder_outfuns, autoencoder_layers,
            autoencoder_convs=autoencoder_convs,
            autoencoder_deconvs=autoencoder_deconvs,
            autoencoder_bn=autoencoder_bn, bn_decay=decay,
            autoencoder_dropouts=autoencoder_dropouts,
            autoencoder_strides=autoencoder_strides) 

    with tf.Session(config=config) as session:
        # writer = tf.summary.FileWriter("output", session.graph)
        session.run(tf.global_variables_initializer())
        R_losses = []
        for epoch in range(epochs):
        
            r_losses = []
            data, data_labels = mmanager.shuffle_imgs()
            for t in range(len(data)//num_samples):
                
                # train step
                curr_data_sample, curr_data_labels = get_data_sample(t)
                curr_data_sample = curr_data_sample.reshape(
                        curr_data_sample.shape[0], img_side, img_side, 1)
                current_decoded_patterns, r_loss = \
                        cae.train_step(session, curr_data_sample, dropout=1.0)
                
                r_losses.append(r_loss)
                
            R_losses.append(np.mean(r_losses))
                
            # test
            curr_decoded_patterns = cae.test_step(session, test_data)
    
            # plot
            plotter.plot(R_losses, test_data[:400], curr_decoded_patterns[:400])
                        
