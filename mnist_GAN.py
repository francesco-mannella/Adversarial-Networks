# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from convolutional_mlp import MLP, linear
from GAN import GAN

"""
Generative adversarial network

* A multilayered perceptron as the generator network - x_z = G(z)
    takes random samples for the latent z vector and gives a 28x28 image x_z
   
* A multilayered perceptron as the discriminator network - p = D(x)
    takes a 28x28 image x and gives a probability that the image belongs 
    to the dataset D (x_d) and not to the generated distribution Z (x_z)

    the discriminator maximizes:
        D_loss = log(D(x_d)) + log(1 - D(G(z)) 
    the generator maximizes:
        G_loss = log( D(G(z)) 
"""

#-------------------------------------------------------------------------------
# only current needed GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#-------------------------------------------------------------------------------
# set the seed for random numbers generation
import os
current_seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]
#current_seed = 1414826039
print "seed:%d" % current_seed
rng = np.random.RandomState(current_seed) 
tf.set_random_seed(current_seed)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# mnist
from mnist_init_training_set import MNIST_manager
mmanager = MNIST_manager()
img_side, n_mnist_pixels, n_train = mmanager.get_params() 
data, data_labels = mmanager.shuffle_imgs()
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# plot
import matplotlib.pyplot as plt
plt.ion()
from mnist_GAN_plot import Plotter
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Globals
epochs = 300
num_samples = 100
lr = 0.00005
dropout = 0.3

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------               
plotter = Plotter(epochs)
#-------------------------------------------------------------------------------   
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


graph = tf.Graph()
with graph.as_default():

    #---------------------------------------------------------------------------
    #  Generator
    generator_layers = [100, 256, 512, 1024, n_mnist_pixels]
    generator_dropouts = [True, True, True, False]
    generator_outfuns =[tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.tanh] 
    
    #---------------------------------------------------------------------------
    #  Disc:q:riminator
    discriminator_layers = [n_mnist_pixels, 1024, 512, 256, 1]
    discriminator_dropouts = [True, True, True, False]
    discriminator_outfuns =[tf.nn.relu, tf.nn.relu, tf.nn.relu,  tf.nn.sigmoid]

    gan = GAN(lr, generator_layers, generator_dropouts, generator_outfuns, 
            discriminator_layers, discriminator_dropouts, discriminator_outfuns)

    #---------------------------------------------------------------------------   
    #---------------------------------------------------------------------------   
    #---------------------------------------------------------------------------     
    # get Samples   
    def get_latent_sample():
        return rng.randn(num_samples, generator_layers[0])
    
    def get_data_sample(iter):
        return np.vstack(data[iter*num_samples:(iter+1)*num_samples])
    
    fixed_latent_sample = get_latent_sample()
    #---------------------------------------------------------------------------   
    #---------------------------------------------------------------------------   
    #---------------------------------------------------------------------------   
  
    # Tf Session
    with tf.Session(config=config) as session:
        # writer = tf.summary.FileWriter("output", session.graph)
        session.run(tf.global_variables_initializer())
        
        HIST_D_losses = []
        HIST_G_losses = []    
        HIST_D_changes = []
        HIST_G_changes = []
        for epoch in range(epochs):
            D_losses = []
            G_losses = []
            D_changes = []
            G_changes = []
            np.random.shuffle(data)
            for iter in range(len(data)//num_samples):
            
                # discriminator step 
                curr_discr_latent_sample = get_latent_sample()
                curr_gen_latent_sample = get_latent_sample()
                curr_data_sample = get_data_sample(iter)
                
                res = gan.train_step(session, curr_discr_latent_sample,
                        curr_gen_latent_sample, curr_data_sample)
                d_probs, d_loss, d_change, g_probs, g_loss, g_change = res  
                
                D_losses.append(d_loss)
                G_losses.append(g_loss)      
                D_changes.append(d_change)
                G_changes.append(g_change)
            
            HIST_D_losses.append(np.mean(D_losses))
            HIST_G_losses.append(np.mean(G_losses))
            HIST_D_changes.append(np.mean(D_changes))
            HIST_G_changes.append(np.mean(G_changes))  
               
            patterns = gan.generative_test_step( session, fixed_latent_sample)

            plotter.plot(
                HIST_G_losses, 
                HIST_D_losses,
                HIST_G_changes, 
                HIST_D_changes, patterns)
                
