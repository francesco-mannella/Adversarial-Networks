# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from convolutional_mlp import MLP, linear
plt.ion()
"""
Generative adversarial autoencoder

* A multilayered perceptron as the autoencoder network 
    takes samples for the mnist dataset and reproduces them
    
    encoder:  X(28*28) -> H1(512) -> H2(256) -> Z(2)
    decoder:  Z(2)     -> H1(256) -> H2(512) -> Y(28*28)
    
* A multilayered perceptron as the discriminator network - p = D(Z)
    takes a sample form the hidden pattern space Z and gives a probability 
    that the image belongs to a prior distribution Z_g
    and not the encoder inner activities Z_d

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
import os
current_seed = np.fromstring(os.urandom(4), dtype=np.uint32)[0]
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
images = np.vstack(mndata.train_images[:])/255.0
images = 2*images - 1
# shuffle img indices
image_labels = np.hstack(mndata.train_labels)
idcs = np.arange(n_train)
def shuffle_imgs():
    np.random.shuffle(idcs)
    return images[idcs], image_labels[idcs]
data, data_labels = shuffle_imgs()
#---------------------------------------------------------------------------  
#---------------------------------------------------------------------------  
#---------------------------------------------------------------------------  
#Globals 
epochs = 100
num_samples = 100
eps = 1e-5
rlr = 0.0005
alr = 0.0005
tests = 10000
#---------------------------------------------------------------------------  
#---------------------------------------------------------------------------  
#---------------------------------------------------------------------------  
# Plot
class Plotter(object):  
    def __init__(self):   
        self.t = 0
        self.n = int(np.sqrt(num_samples))

        self.fig = plt.figure(figsize=(12,8))
        l = 6
        gs = gridspec.GridSpec(10 + l, 10*2)
        
        self.losses_ax = self.fig.add_subplot(gs[:l-1,:])
        self.losses_ax.set_title("Reconstruction error")
        self.losses_lines = []
        line, = self.losses_ax.plot(0,0)   
        self.losses_lines.append(line)
        self.labels = ["reconstruction"]  
        self.losses_ax.legend(self.losses_lines, self.labels)  
        self.losses_ax.set_xlim([0, epochs]) 
        self.losses_ax.set_ylim([0.1, 0.4])
        self.losses_ax.set_yticks(np.linspace(0,0.4,11))
        
        self.losses_ax.grid(color='b', linestyle='--', linewidth=0.1)
                   
        self.hidden_ax = self.fig.add_subplot(gs[l:,:10])
        self.hidden_ax.set_title("Hidden layer activation")
        self.hidden_scatter = self.hidden_ax.scatter(0,0, lw=0, s=5)
        self.hidden_ax.set_xlim([-5,5])
        self.hidden_ax.set_ylim([-5,5])
        
        self.pattern_axes = []
        self.pattern_imgs = []
        for x in range(10):
            for y in range(10):
                ax = self.fig.add_subplot(gs[l + x, 10 + y])
                im = ax.imshow(np.zeros([img_side, img_side]), 
                               vmin=-1, vmax=1)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                self.pattern_axes.append(ax)
                self.pattern_imgs.append(im)
        plt.subplots_adjust(bottom=0.1,top=0.9, 
                            left=0.1, right=0.9, 
                            hspace=0.0, wspace=0.0)
        
        
    def plot(self, R_loss, hidden, labels, patterns):
        
        losses = [R_loss]   
        t = len(R_loss)
        self.losses_lines[0].set_data(np.arange(t), R_loss)
            
        self.hidden_scatter.set_offsets(hidden)   
        self.hidden_scatter.set_facecolor(plt.cm.hsv(labels/10.0))    
        self.hidden_scatter.set_edgecolor(plt.cm.hsv(labels/10.0))    
        
        for x in range(10):
            for y in range(10):
                k = x*10 + y     
                im = self.pattern_imgs[k]
                im.set_data(patterns[k].reshape(img_side, img_side))
        self.fig.canvas.draw()
        self.fig.savefig("aae-{:03d}.png".format(self.t))
        self.t += 1
                 
plotter = Plotter()
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def get_data_sample(t):
    """
    get a batch sample of mnist patterns 
    :param t: the binibatch step within an epoch 
    """
    sample_rng = range(t*num_samples,(t+1)*num_samples)
    return data[sample_rng].reshape(num_samples, img_side, img_side, 1), data_labels[sample_rng]     
#-------------------------------------------------------------------------------
def get_simple_prior_sample():
    """
    get a batch sample of hidden pattern pairs generated from a 
    simple N(0, 1) distribution
    """
    return rng.randn(num_samples, 2)
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

    res = np.vstack([
        np.random.multivariate_normal(
                        [1.8*np.sin(k), 1.8*np.cos(k)],
                        sigma(theta=k, sigma_x=0.2, sigma_y=2.0),
                        num_samples/10) 
                      for k in np.linspace(0, (2*np.pi)*(9./10.), 10)])
    return res
#-------------------------------------------------------------------------------
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# pointer to the chosen prior distribution 
get_prior_sample = get_10_prior_sample
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#-------------------------------------------------------------------------------
def get_grid():  
    """
    Define a distributed grid of points in the space of 
    the hidden patterns pair
    """        
    x = np.linspace(-2.0, 2.0, 10)     
    X, Y = np.meshgrid(x,x)
    return np.vstack((X.ravel(), Y.ravel())).T
# build the grid 
grid = get_grid()
#-------------------------------------------------------------------------------
# data used for the test phase
test_data = data[:tests].reshape(tests, img_side, img_side, 1)
test_labels = data_labels[:tests]
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
graph = tf.Graph()
with graph.as_default():
#-------------------------------------------------------------------------------
    drop_out = tf.placeholder(tf.float32, ())
    #  Encoder
    with tf.variable_scope('Encoder'):
        
        #input
        encoder_layers     = [ [28, 28, 1]  ]

        # encoder
        encoder_layers    += [ [14, 14, 3],    [7, 7, 5],         [7, 7, 10],       [2],           ]
        encoder_outfuns    = [ tf.nn.relu,     tf.nn.relu,        tf.nn.relu,       linear         ] 
        encoder_convs      = [ [3, 3, 1, 3],   [12, 12, 3, 5],    [20, 20, 5, 10],  None           ]
        encoder_deconvs    = [ None,           None,              None,             None           ]
        encoder_strides    = [ 2,              2,                 1,                None           ]
          
        encoder = MLP(lr=rlr, scope="E",
                             outfuns=encoder_outfuns, 
                             layers_lens=encoder_layers,
                             convs=encoder_convs,
                             deconvs=encoder_deconvs,
                             strides=encoder_strides)  
         
        data_sample = tf.placeholder(tf.float32, [num_samples] + encoder_layers[0])
        hidden_patterns = encoder.update(data_sample, drop_out)   
        #------
        data_test = tf.placeholder(tf.float32, [tests] + encoder_layers[0])
        test_hidden_patterns = encoder.update(data_test, drop_out)      
    #---------------------------------------------------------------------------
    #  Decoder
    with tf.variable_scope('decoder'):
        
        #input
        decoder_layers    = [ [2] ]     
        
        # decoder
        decoder_layers   += [ [7, 7, 10],     [7, 7, 5],         [14, 14, 3],      [28, 28, 1]    ]
        decoder_outfuns   = [ tf.nn.relu,     tf.nn.relu,        tf.nn.relu,       tf.tanh        ] 
        decoder_convs     = [ None,           None,              None,             None           ]
        decoder_deconvs   = [ None,           [20, 20, 5, 10],   [12, 12, 3, 5],   [3, 3, 1, 3]   ]
        decoder_strides   = [ None,           1,                 2,                2              ]
        
        decoder = MLP(lr=alr, scope="D",
                             outfuns=decoder_outfuns, 
                             layers_lens=decoder_layers,
                             convs=decoder_convs,
                             deconvs=decoder_deconvs,
                             strides=decoder_strides)  
          
        decoded_patterns = decoder.update(hidden_patterns, drop_out)   
    #---------------------------------------------------------------------------
    #  Adversarial       
    with tf.variable_scope('adversarial'):
        
        #input
        adversarial_layers     = [ [2]  ]     
        
        # adversarial
        adversarial_layers    += [ [7, 7, 10],     [7, 7, 5],         [14, 14, 3],      [1]            ]
        adversarial_outfuns    = [ tf.nn.relu,     tf.nn.relu,        tf.nn.relu,       tf.sigmoid     ] 
        adversarial_convs      = [ None,           None,              None,             None           ]
        adversarial_deconvs    = [ None,           [20, 20, 5, 10],   [12, 12, 3, 5],   None           ]
        adversarial_strides    = [ None,           1,                 2,                None           ]
        
        adversarial = MLP(lr=rlr, scope="A",
                             outfuns=adversarial_outfuns, 
                             layers_lens=adversarial_layers,
                             convs=adversarial_convs,
                             deconvs=adversarial_deconvs,
                             strides=adversarial_strides)   
        
        prior_sample = tf.placeholder(tf.float32, [num_samples] + decoder_layers[0])
        test_sample = tf.placeholder(tf.float32, [100] + decoder_layers[0])
        generated_patterns = decoder.update(prior_sample, drop_out)   
        test_generated_patterns = decoder.update(test_sample, drop_out)   
        D_probs = adversarial.update(prior_sample, drop_out)
        G_probs = adversarial.update(hidden_patterns, drop_out)        
    #---------------------------------------------------------------------------   
    R_loss = tf.losses.mean_squared_error(data_sample, decoded_patterns)
    D_loss = tf.reduce_mean(-tf.log(D_probs + eps) - tf.log(1.0 - G_probs + eps))  
    G_loss = tf.reduce_mean(-tf.log(G_probs + eps))
    #---------------------------------------------------------------------------
    ER_train =  encoder.train(R_loss, lr=rlr)
    DR_train =  decoder.train(R_loss, lr=rlr)
    D_train =  adversarial.train(D_loss, lr=alr)
    G_train =  encoder.train(G_loss, lr=alr)
    #---------------------------------------------------------------------------   
    #---------------------------------------------------------------------------   
    #---------------------------------------------------------------------------   
    # Tf Session
    with tf.Session(config=config) as session:
        writer = tf.summary.FileWriter("output", session.graph)
        session.run(tf.global_variables_initializer())
        R_losses = []
        for epoch in range(epochs):
        
            r_losses = []
            data, data_labels = shuffle_imgs()
            for t in range(len(data)//num_samples):
            
                # reconstruction step -- encoder -> decoder (minimize reconstruction  error)
                curr_data_sample, curr_data_labels = get_data_sample(t)
                current_decoded_patterns, r_loss, _, _= session.run(
                    [decoded_patterns, R_loss, ER_train, DR_train], 
                    feed_dict={data_sample:curr_data_sample, 
                               drop_out: 0.3})  
                
                # adversarial step -- prior -> adversarial (minimize discrimination error)
                curr_prior_sample = get_prior_sample()
                d_loss, _ = session.run([D_loss, D_train], 
                    feed_dict={data_sample:curr_data_sample, 
                               prior_sample:curr_prior_sample, 
                               drop_out: 0.3})               
                 
                # adversarial step -- hidden -> adversarial (minimize discrimination error)
                curr_prior_sample = get_prior_sample()
                g_loss, _ = session.run([G_loss, G_train], 
                    feed_dict={data_sample:curr_data_sample, 
                               drop_out: 0.3})                
                 
                r_losses.append(r_loss)
                
            R_losses.append(np.mean(r_losses))
            
            curr_patterns = session.run(test_generated_patterns, 
                feed_dict={test_sample:grid,  drop_out: 1.0})
            curr_hidden_patterns = session.run(test_hidden_patterns, 
                feed_dict={data_test:test_data,  drop_out: 1.0})
            
            plotter.plot(R_losses, curr_hidden_patterns, test_labels, curr_patterns)
                
        raw_input()