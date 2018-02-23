# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from convolutional_mlp import MLP, linear, leaky_relu
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
from mnist_ACAE_plot import Plotter
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
#---------------------------------------------------------------------------  
#---------------------------------------------------------------------------  
#---------------------------------------------------------------------------  
#Globals 
epochs = 100
num_samples = 100
eps = 1e-10
rlr = 0.0001
alr = 0.8
tests = 60000
dropout_train = 0.8
dropout_test = 1.0
weight_scale = 0.02
decay = 0.1

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
test_data = data[:tests].reshape(tests, img_side, img_side, 1)
test_labels = data_labels[:tests]
#-------------------------------------------------------------------------------
print "globals and samples initialized"
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
graph = tf.Graph()
with graph.as_default():
#-------------------------------------------------------------------------------
    drop_out = tf.placeholder(tf.float32, ())
    phase_train = tf.placeholder(tf.bool)
    
    #---------------------------------------------------------------------------
    # Encoder
    #### network init 
    #input
    encoder_layers     = [ [28, 28, 1]  ]
    # encoder
    encoder_layers    += [ [14, 14, 3],    [7, 7, 25],        [7, 7, 10],       [2],           ]
    encoder_outfuns    = [ tf.nn.relu,     tf.nn.relu,        tf.nn.relu,       linear         ] 
    encoder_convs      = [ [3, 3, 1, 3],   [12, 12, 3, 25],   [20, 20, 25, 10], None           ]
    encoder_deconvs    = [ None,           None,              None,             None           ]
    encoder_strides    = [ 2,              2,                 1,                None           ]
    encoder_bn         = [ True,           True,              True,             False          ]
    
    encoder = MLP(scope="Encoder", 
                  lr=rlr,
                  bn_decay=decay,
                  weight_scale=weight_scale,
                  outfuns=encoder_outfuns, 
                  convs=encoder_convs, 
                  deconvs=encoder_deconvs, 
                  strides=encoder_strides,
                  batch_norms=encoder_bn,
                  layers_lens=encoder_layers)   

    data_sample = tf.placeholder(tf.float32, [num_samples] + encoder_layers[0])
    #### training branch
    hidden_patterns = encoder.update(data_sample, drop_out=drop_out, phase_train=phase_train)
    #### test branch
    data_test = tf.placeholder(tf.float32, [tests] + encoder_layers[0])
    test_hidden_patterns = encoder.update(data_test, drop_out=drop_out, phase_train=phase_train)      
    print "Encoder tree has been built"

    #---------------------------------------------------------------------------
    # Decoder
    #### network init
    #input
    decoder_layers    = [ [2] ]     
    # decoder
    decoder_layers   += [ [7, 7, 10],     [7, 7, 25],        [14, 14, 3],      [28, 28, 1]    ]
    decoder_outfuns   = [ tf.nn.relu,     tf.nn.relu,        tf.nn.relu,       tf.tanh        ] 
    decoder_convs     = [ None,           None,              None,             None           ]
    decoder_deconvs   = [ None,           [20, 20, 25, 10],  [12, 12, 3, 25],  [3, 3, 1, 3]   ]
    decoder_bn        = [ False,          True,              True,             True           ]
    decoder_strides   = [ None,           1,                 2,                2              ]
        
    decoder = MLP(scope="Decoder",
            lr=rlr, 
            bn_decay=decay,
            weight_scale=weight_scale,
            outfuns=decoder_outfuns, 
            layers_lens=decoder_layers,
            convs=decoder_convs,
            deconvs=decoder_deconvs,
            batch_norms=decoder_bn,
            strides=decoder_strides)  

    #### training branch
    decoded_patterns = decoder.update(hidden_patterns, drop_out=drop_out, phase_train=phase_train)   
    print "Decoder tree has been built"

    #---------------------------------------------------------------------------
    #  Adversarial       
    #### network init
    #input
    adversarial_layers     = [ [2]  ]     
    # adversarial
    adversarial_layers    += [ [1000],          [1000],           [1]            ]
    adversarial_outfuns    = [ tf.nn.relu,      tf.nn.relu,       tf.sigmoid     ] 
    adversarial_convs      = [ None,            None,             None           ]
    adversarial_deconvs    = [ None,            None,             None           ]
    adversarial_strides    = [ None,            None,             None           ]
    adversarial_dropout    = [ True,            True,             False          ]
    adversarial_bn         = [ False,           False,            False          ]

    adversarial = MLP(scope="Adversarial",
            lr=alr, 
            bn_decay=decay,
            weight_scale=weight_scale,
            outfuns=adversarial_outfuns, 
            layers_lens=adversarial_layers,
            convs=adversarial_convs,
            deconvs=adversarial_deconvs,
            strides=adversarial_strides,
            batch_norms=adversarial_bn,
            drop_out=adversarial_dropout)   

    #### Discriminator branch
    prior_sample = tf.placeholder(tf.float32, [num_samples] + decoder_layers[0])
    prior_generated_patterns = decoder.update(prior_sample, drop_out=drop_out, phase_train=phase_train)   
    D_probs = adversarial.update(prior_sample, drop_out=drop_out, phase_train=phase_train)
    #### Generator branch
    G_probs = adversarial.update(hidden_patterns, drop_out=drop_out, phase_train=phase_train)        
    print "Adversarial tree has been built"
    test_sample = tf.placeholder(tf.float32, [400] + decoder_layers[0])
    test_generated_patterns = decoder.update(test_sample, drop_out=drop_out, phase_train=phase_train)  

    #---------------------------------------------------------------------------   
    # Losses
    R_loss = tf.losses.mean_squared_error(data_sample, decoded_patterns)
    D_loss = tf.reduce_mean(-tf.log(D_probs + eps) - tf.log(1.0 - G_probs + eps))  
    G_loss = tf.reduce_mean(-tf.log(G_probs + eps))
    print "Losses branches have been added"

    #---------------------------------------------------------------------------
    # Optimizations
    ER_train =  encoder.train(R_loss, lr=rlr)
    DR_train =  decoder.train(R_loss, lr=rlr)
    D_train =  adversarial.train(D_loss, lr=alr)
    G_train =  encoder.train(G_loss, lr=alr)
    print "Optimizers branches have been added"

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
            data, data_labels = mmanager.shuffle_imgs()
            for t in range(len(data)//num_samples):
                
                # reconstruction step -- (minimize reconstruction  error)
                #    data_sample -> encoder -> hidden_patterns -> decoder -> decoded_patterns
                curr_data_sample, curr_data_labels = get_data_sample(t)
                curr_data_sample = curr_data_sample.reshape(num_samples, img_side, img_side, 1)
                current_decoded_patterns, r_loss, _, _= session.run(
                    [decoded_patterns, R_loss, ER_train, DR_train], 
                    feed_dict={data_sample:curr_data_sample, 
                               phase_train: True})  
                
                # adversarial step -- (minimize discrimination error)
                #    data_sample  -> encoder -> hidden_patterns -> adversarial -> G_props
                #    prior_sample                               -> adversarial -> D_props
                curr_prior_sample = get_prior_sample()
                d_loss, _ = session.run([D_loss, D_train], 
                    feed_dict={data_sample:curr_data_sample, 
                               prior_sample:curr_prior_sample, 
                               drop_out: dropout_train, phase_train: True})               
                
                # adversarial step -- (maximize discrimination error)
                #    data_sample  -> encoder -> hidden_patterns -> adversarial -> G_props
                curr_prior_sample = get_prior_sample()
                g_loss, _ = session.run([G_loss, G_train], 
                    feed_dict={data_sample:curr_data_sample, 
                               drop_out: dropout_train, phase_train: True})                
                
                r_losses.append(r_loss)
                
            R_losses.append(np.mean(r_losses))
           
            # test
            # Current generation of image patterns in response to 
            #   the presentation of a 2D grid of values to the two hidden units 
            curr_patterns = session.run(test_generated_patterns, 
                feed_dict={test_sample:grid, phase_train: False})
            # Current activation of hidden units in response to the presentation of 
            #   60000 data images
            curr_hidden_patterns = session.run(test_hidden_patterns, 
                feed_dict={data_test:test_data, phase_train: False})
            


            # plot
            plotter.plot(R_losses, curr_hidden_patterns, test_labels, curr_patterns)
                        
