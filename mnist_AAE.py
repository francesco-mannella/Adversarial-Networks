# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from convolutional_mlp import linear, leaky_relu
from AAE import AAE
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
from mnist_AAE_plot import Plotter
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
"""
Adversarial autoencoder

* A multilayered perceptron as the autoencoder network 
    takes samples for the mnist dataset and reproduces them
    
    encoder:  X(28*28) -> H1(1000) -> H2(1000) -> Z(2)
    decoder:  Z(2)     -> H1(1000) -> H2(1000) -> Y(28*28)
    
* A multilayered perceptron as the discriminator network - p = D(Z)
    takes a sample form the hidden pattern space Z and gives a probability 
    that the image belongs to a prior distribution Z_g
    and not the encoder inner activities Z_d
    
    discriminator:  Z(2) -> H1(1000) -> H2(1000) -> Y(1)

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
num_samples = 50
eps = 1e-10
rlr = 0.0001
alr = 0.0001
tests = 60000
dropout_train = 0.2
dropout_test = 1.0
weight_scale = 0.02
decay = 0.01
plotter = Plotter(epochs)

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
#-------------------------------------------------------------------------------
def get_simple_prior_sample():
    """
    get a batch sample of hidden pattern pairs generated from a 
    simple N(0, 1) distribution
    """
    return 10*rng.randn(num_samples, 2)
#-------------------------------------------------------------------------------
def get_2_prior_sample():
    """
    get a batch sample of hidden patterns pairs generated from a 
    mixture of N(m1, s1) and N(m2, s2) 
    """
    s1 = np.hstack((.2, .2)).reshape(1,2)
    s2 = np.hstack((.2, .2)).reshape(1,2)
    m1 = np.hstack((-.5, .5)).reshape(1,2)
    m2 = np.hstack((.5, -.5)).reshape(1,2)
    switch = rng.rand(num_samples) > 0.5
    switch = switch.reshape(num_samples, 1)
    return switch*(s1*rng.randn(num_samples, 2) + m1) + \
        (1-switch)*(s2*rng.randn(num_samples, 2) + m2)

thetas = np.linspace(0, (2*np.pi)*(9./10.), 10) 
def get_10_prior_sample():
    """
    get a batch sample of hidden patterns pairs generated from a 
    mixture of ten gaussian placed radially with respect to the origin  
    """
    def sigma(theta=np.pi, sigma_x=1, sigma_y=2):
        sigma_x = 1.0/sigma_x
        sigma_y = 1.0/sigma_y
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -np.sin(2*theta)/(4*sigma_x**2) + np.sin(2*theta)/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        return np.array([[a, b],[b, c]])

    curr_thetas = thetas[rng.randint(0, 10, num_samples)]
        
    res = np.vstack([
        rng.multivariate_normal(
                        [12*np.sin(k), 12*np.cos(k)],
                        sigma(theta=k, sigma_x=1, sigma_y=10), 1) 
                      for k in curr_thetas])
    return res

# pointer to the chosen prior distribution 
get_prior_sample = get_10_prior_sample

#-------------------------------------------------------------------------------

def get_grid():  
    """
    Define a distributed grid of points in the space of 
    the hidden patterns pair
    """        
    x = np.linspace(-40, 40, 20)     
    Y, X = np.meshgrid(x,x)
    res = np.vstack((X.ravel(), Y.ravel())).T
    res = res[:, ::-1]
    return res

# build the grid 
grid = get_grid()

#-------------------------------------------------------------------------------

# data used for the test phase
test_data = data[:tests]
test_labels = data_labels[:tests]

#-------------------------------------------------------------------------------
print "globals and samples initialized"
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

graph = tf.Graph()
with graph.as_default():
    
    encoder_layers = [n_mnist_pixels, 1000, 1000, 2]
    encoder_dropout =[False, False, False] 
    encoder_outfuns =[tf.nn.relu, tf.nn.relu, linear] 

    decoder_layers = [2, 1000, 1000, n_mnist_pixels]
    decoder_dropout =[False, False, False] 
    decoder_outfuns =[tf.nn.relu, tf.nn.relu, tf.nn.tanh] 

    adversarial_layers = [2, 1000, 1000, 1]
    adversarial_dropout =[True, True, False] 
    adversarial_outfuns =[ tf.nn.relu, tf.nn.relu, tf.nn.sigmoid] 

    aae = AAE(rlr, alr, weight_scale, 
            encoder_outfuns, encoder_layers,
            decoder_outfuns, decoder_layers,
            adversarial_outfuns, adversarial_layers)

    # Tf Session
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
                curr_prior_sample = get_prior_sample()
                current_decoded_patterns, r_loss, d_loss, g_loss = \
                        aae.train_step(session, curr_data_sample, curr_prior_sample)
                
                r_losses.append(r_loss)
                
            R_losses.append(np.mean(r_losses))
                
            # test
            curr_patterns, curr_hidden_patterns = aae.test_step(session, grid, test_data)
    
            # plot
            plotter.plot(R_losses, curr_hidden_patterns, test_labels, curr_patterns)
                        
