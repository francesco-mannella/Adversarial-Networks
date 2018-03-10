# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from convolutional_mlp import linear, tanh, leaky_relu
from ACAE import AAE
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
from mnist_ACAE_plot import Plotter
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
num_samples = 50
eps = 1e-5
rlr = 0.0005
alr = 0.0005
tests = 60000
dropout_train = 0.3
dropout_test = 1.0
weight_scale = 0.1
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
def get_10_prior_sample(num_samples=num_samples):
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
                        sigma(theta=k, sigma_x=2, sigma_y=11), 1) 
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
prior_pop = np.vstack([get_prior_sample() for x in range(200)])

plotter = Plotter(epochs, prior_pop)
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
    
    encoder_layers      = [[28, 28, 1]]
    encoder_layers     += [[14, 14, 30],  [7, 7, 15],       [7, 7, 5],         [200],              [2]]
    encoder_convs       = [[3, 3, 1, 30], [3, 3, 30, 15],   [7, 7, 15, 5],     None,               None]
    encoder_deconvs     = [None,          None,             None,              None,               None]
    encoder_copy_from   = [None,          None,             None,              None,               None]
    encoder_strides     = [2,             2,                1,                 None,               None]
    encoder_outfuns     = [tf.nn.relu,    tf.nn.relu,       tf.nn.relu,        tf.nn.relu,         linear]
    encoder_bn          = [False,         True,             True,              True,               False] 
    encoder_dropouts    = [False,         False,            False,             False,              False] 
    
    decoder_layers      = [[2]]
    decoder_layers     += [[200],         [7, 7, 5],        [7, 7, 15],        [14, 14, 30],       [28, 28, 1]]
    decoder_convs       = [None,          None,             None,              None,               None]
    decoder_deconvs     = [None,          None,             [7, 7, 15, 5],     [3, 3, 30, 15],     [3, 3, 1, 30]]
    decoder_copy_from   = [None,          None,             None,              None,               None]
    decoder_strides     = [None,          None,             1,                 2,                  2]
    decoder_outfuns     = [tf.nn.relu,    tf.nn.relu,       tf.nn.relu,        tf.nn.relu,         linear] 
    decoder_bn          = [False,         True,             True,              True,               False] 
    decoder_dropouts    = [False,         False,            False,             False,              False]  
    

    adversarial_layers      = [[2]]
    adversarial_layers     += [[14, 14, 30],  [7, 7, 15],       [7, 7, 5],         [200],              [1]]
    adversarial_convs       = [[3, 3, 1, 30], [3, 3, 30, 15],   [7, 7, 15, 5],     None,               None]
    adversarial_deconvs     = [None,          None,             None,              None,               None]
    adversarial_copy_from   = [None,          None,             None,              None,               None]
    adversarial_strides     = [2,             2,                1,                 None,               None]
    adversarial_outfuns     = [tf.nn.relu,    tf.nn.relu,       tf.nn.relu,        tf.nn.relu,         linear]
    adversarial_bn          = [False,         True,             True,              True,               False] 
    adversarial_dropouts    = [False,         False,            False,             False,              False] 

    # adversarial_layers     += [[1000],        [1000],        [1]]
    # adversarial_convs       = [None,          None,          None]
    # adversarial_deconvs     = [None,          None,          None]
    # adversarial_copy_from   = [None,          None,          None]
    # adversarial_strides     = [None,          None,          None]
    # adversarial_outfuns     = [tf.nn.relu,    tf.nn.relu,    linear]
    # adversarial_bn          = [False,         False,         False] 
    # adversarial_dropouts    = [True,          True,          False] 

    aae = AAE(rlr, alr, weight_scale, 
            encoder_outfuns, encoder_layers,
            decoder_outfuns, decoder_layers,
            adversarial_outfuns, adversarial_layers, 
            encoder_convs=encoder_convs, 
            encoder_bn=encoder_bn, bn_decay=decay,
            encoder_dropouts=encoder_dropouts,
            encoder_strides=encoder_strides,
            decoder_deconvs=decoder_deconvs, 
            decoder_bn=decoder_bn, 
            decoder_dropouts=decoder_dropouts,
            decoder_strides=decoder_strides,
            adversarial_deconvs=adversarial_deconvs, 
            adversarial_bn=adversarial_bn, 
            adversarial_dropouts=adversarial_dropouts,
            adversarial_strides=adversarial_strides
            )

    aae.eps = eps

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
                curr_data_sample = curr_data_sample.reshape( 
                        curr_data_sample.shape[0],
                        img_side, img_side, 1)
                curr_prior_sample = get_prior_sample(num_samples*20)
                current_decoded_patterns, r_loss, d_loss, g_loss = \
                        aae.train_step(session, curr_data_sample, curr_prior_sample)
                
                r_losses.append(r_loss)
                
            R_losses.append(np.mean(r_losses))
                
            # test
            curr_generated_patterns, curr_hidden_patterns, \
                    curr_decoded_patterns = aae.test_step(session, 
                            grid, test_data)
    
            # plot
            plotter.plot(R_losses, curr_hidden_patterns, test_labels,
                    curr_generated_patterns, curr_decoded_patterns[:400])
                        
