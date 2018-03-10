# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from convolutional_mlp import MLP, linear, leaky_relu, tf_shape
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# mnist
from mnist_init_training_set import MNIST_manager
mmanager = MNIST_manager()
img_side, n_mnist_pixels, n_train = mmanager.get_params() 
data, data_labels = mmanager.order_imgs()
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
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
epochs = 1000
num_samples = 50
eps = 1e-10
rlr = 0.0004
alr = 0.0004
tests = 60000
dropout_train = 0.2
dropout_test = 1.0
weight_scale = 0.06
decay = 0.01

plotter = Plotter(epochs)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def add_label(x, label, depth=11):
    label = tf.one_hot(label, depth)
    label = tf.squeeze(label)
    x_labelled = tf.concat([x, tf.cast(label, tf.float32)], axis=1)   
    return x_labelled
    
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

    lbls = rng.randint(0, 10, num_samples)
    curr_thetas = thetas[lbls]
        
    res = np.vstack([
        rng.multivariate_normal(
                        [12*np.sin(k), 12*np.cos(k)],
                        sigma(theta=k, sigma_x=1, sigma_y=10), 1) 
                      for k in curr_thetas])
    return res, lbls

def get_10_prior_sample_labelled(label):
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

    lbls = label*np.ones(num_samples).astype(int)
    curr_thetas = thetas[lbls]
        
    res = np.vstack([
        rng.multivariate_normal(
                        [12*np.sin(k), 12*np.cos(k)],
                        sigma(theta=k, sigma_x=1, sigma_y=10), 1) 
                      for k in curr_thetas])
    return res, lbls
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
#-------------------------------------------------------------------------------
    drop_out = tf.placeholder(tf.float32, ())
    phase_train = tf.placeholder(tf.bool)
    
    #---------------------------------------------------------------------------
    # Encoder
    #### network init 
    encoder_layers = [n_mnist_pixels, 1000, 1000, 2]
    encoder_dropout =[False, False, False] 
    encoder_bn = [False, False, False] 
    encoder_outfuns =[tf.nn.relu, tf.nn.relu, linear] 
    encoder = MLP(scope="Encoder", 
                  lr=rlr,
                  bn_decay=decay,
                  weight_scale=weight_scale,
                  batch_norms=encoder_bn,
                  outfuns=encoder_outfuns, 
                  drop_out=encoder_dropout,
                  layers_lens=encoder_layers)   
    data_sample = tf.placeholder(tf.float32, [num_samples, encoder_layers[0]])
    #### training branch
    hidden_patterns = encoder.update(data_sample, drop_out=drop_out, phase_train=phase_train)   
    #### test branch
    data_test = tf.placeholder(tf.float32, [tests, encoder_layers[0]])
    test_hidden_patterns = encoder.update(data_test, drop_out=drop_out, phase_train=phase_train)      
    print "Encoder tree has been built"
    
    #---------------------------------------------------------------------------
    # Decoder
    #### network init
    decoder_layers = [2, 1000, 1000, n_mnist_pixels]
    decoder_dropout =[False, False, False] 
    decoder_bn = [False, False, False] 
    decoder_outfuns =[tf.nn.relu, tf.nn.relu, tf.nn.tanh] 
    decoder = MLP(scope="Decoder",
                  lr=rlr, 
                  bn_decay=decay,
                  weight_scale=weight_scale,
                  batch_norms=decoder_bn,
                  drop_out=decoder_dropout,
                  outfuns=decoder_outfuns, 
                  layers_lens=decoder_layers)    
    #### training branch
    decoded_patterns = decoder.update(hidden_patterns, drop_out=drop_out, phase_train=phase_train)   
    print "Decoder tree has been built"
    
    #---------------------------------------------------------------------------
    #  Adversarial       
    #### network init
    adversarial_layers = [13, 1000, 1000, 1]
    adversarial_dropout =[True, True, False] 
    adversarial_bn = [False, False, False]
    adversarial_outfuns =[ tf.nn.relu, tf.nn.relu, tf.nn.sigmoid] 
    adversarial = MLP(scope="Adversarial",
                      lr=alr, 
                      bn_decay=decay,
                      weight_scale=weight_scale,
                      batch_norms=adversarial_bn,
                      drop_out=adversarial_dropout,
                      outfuns=adversarial_outfuns,
                      layers_lens=adversarial_layers)   
    #### Discriminator branch
    prior_sample = tf.placeholder(tf.float32, [num_samples, decoder_layers[0]])
    prior_labels = tf.placeholder(tf.int32, [num_samples])
    prior_generated_patterns = decoder.update(prior_sample, drop_out=drop_out, phase_train=phase_train)   
    prior_sample_labelled = add_label(prior_sample, tf.cast(prior_labels, tf.int32))
    D_probs = adversarial.update(prior_sample_labelled, drop_out=drop_out, phase_train=phase_train)
    #### Generator branch
    labels = tf.placeholder(tf.int32, [num_samples])
    hidden_patterns_labelled = add_label(hidden_patterns, labels)
    G_probs = adversarial.update(hidden_patterns_labelled, drop_out=drop_out, phase_train=phase_train)        
    print "Adversarial tree has been built"
    #### test branch
    test_sample = tf.placeholder(tf.float32, [400, decoder_layers[0]])
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
        
        curr_label = rng.randint(0,10) 
        for epoch in range(epochs):
            if epoch % 3 == 0: curr_label = rng.randint(0,10) 
            r_losses = []
            data, data_labels = mmanager.labelled_imgs(curr_label)
            for t in range(len(data)//num_samples):
                
                # reconstruction step -- (minimize reconstruction  error)
                #    data_sample -> encoder -> hidden_patterns -> decoder -> decoded_patterns
                curr_data_sample, curr_data_labels = get_data_sample(t)
                current_decoded_patterns, r_loss, _, _= session.run(
                    [decoded_patterns, R_loss, ER_train, DR_train], 
                    feed_dict={data_sample:curr_data_sample, 
                               drop_out: dropout_test, phase_train: True})  
                
                # adversarial step -- (minimize discrimination error)
                #    data_sample  -> encoder -> hidden_patterns -> adversarial -> G_props
                #    prior_sample                               -> adversarial -> D_props
                curr_prior_sample, curr_prior_labels = get_10_prior_sample_labelled(curr_label)
                d_loss, _ = session.run([D_loss, D_train], 
                    feed_dict={data_sample:curr_data_sample, 
                               labels:curr_data_labels,
                               prior_sample:curr_prior_sample, 
                               prior_labels:curr_prior_labels,
                               drop_out: dropout_train, phase_train: True})               
                
                # adversarial step -- (maximize discrimination error)
                #    data_sample  -> encoder -> hidden_patterns -> adversarial -> G_props
                g_loss, _ = session.run([G_loss, G_train], 
                    feed_dict={data_sample:curr_data_sample,
                               labels: curr_data_labels,
                               drop_out: dropout_train, phase_train: True})                
                
                r_losses.append(r_loss)
                
            R_losses.append(np.mean(r_losses))
           
            # test
            # Current generation of image patterns in response to 
            #   the presentation of a 2D grid of values to the two hidden units 
            curr_patterns = session.run(test_generated_patterns, 
                feed_dict={test_sample:grid, drop_out: dropout_test, phase_train: False})
            # Current activation of hidden units in response to the presentation of 
            #   10000 data images
            curr_hidden_patterns = session.run(test_hidden_patterns, 
                feed_dict={data_test:test_data, drop_out: dropout_test, phase_train: False})
    
            # plot
            plotter.plot(R_losses, curr_hidden_patterns, test_labels, curr_patterns)
                        
