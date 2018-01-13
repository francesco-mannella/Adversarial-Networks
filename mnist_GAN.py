# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from mlp import MLP, linear
plt.ion()

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
n_train = len(mndata.train_images[:6000])
# convert data into {-1,1} arrays
data = np.vstack(mndata.train_images[:6000])/255.0
data = 2*data - 1
np.random.shuffle(data)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Globals
epochs = 300
num_samples = 50
eps = 1e-20
lr = 0.0002
dropout = 0.6
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Plot
class Plotter(object):  
    def __init__(self):
        
        self.t = 0
        self.fig = plt.figure(figsize=(10,6))
        gs = gridspec.GridSpec(5, 10)

        self.changes_ax = self.fig.add_subplot(gs[:2, :5])
        self.changes_ax.set_title("Weight change")
        self.cd_line, = self.changes_ax.plot(0,0)
        self.cg_line, = self.changes_ax.plot(0,0)
        self.changes_ax.legend([self.cd_line, self.cg_line], 
                              ["discriminator", "generator"])  
        self.changes_ax.set_xlim([0, epochs]) 
        self.changes_ax.set_ylim([0, 3.5])
          
        self.losses_ax = self.fig.add_subplot(gs[2:4, :5])
        self.losses_ax.set_title("Losses")
        self.ld_line, = self.losses_ax.plot(0,0)
        self.lg_line, = self.losses_ax.plot(0,0)
        self.losses_ax.legend([self.ld_line, self.lg_line], 
                ["discriminator: log(D(x)) + log(1 - D(G(z)))", 
                    "generator: log(D(G(z)))"])  
        self.losses_ax.set_xlim([0, epochs]) 
        self.losses_ax.set_ylim([0, -6])
          
        self.pattern_axes = []
        self.pattern_imgs = []
        for x in range(5):
            for y in range(5):
                ax = self.fig.add_subplot(gs[x, 5+y])
                im = ax.imshow(np.zeros([img_side, img_side]), 
                               vmin=-1, vmax=1)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                self.pattern_axes.append(ax)
                self.pattern_imgs.append(im)

    def plot(self, G_losses, D_losses, G_changes, D_changes, patterns):
          
        t = range(len(G_losses))

        self.ld_line.set_data(t, D_losses)
        self.lg_line.set_data(t, G_losses)
        
        self.cd_line.set_data(t, D_changes)
        self.cg_line.set_data(t, G_changes)
                                    
        l = len(patterns)
        for x in range(5):
            for y in range(5):
                k = x*5 + y     
                im = self.pattern_imgs[k]
                if k<l:
                    im.set_data(patterns[k].reshape(img_side, img_side))
        plt.tight_layout(pad=0.1)
        self.fig.canvas.draw()
        self.fig.savefig("gan-last.png")
        self.fig.savefig("gan-{:03d}.png".format(self.t))
        self.t += 1
                 
plotter = Plotter()
#-------------------------------------------------------------------------------   
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
graph = tf.Graph()
with graph.as_default():
    #---------------------------------------------------------------------------
    #  Generator
    with tf.variable_scope('Generator'):
        generator_layers = [100, 256, 512, 1024, n_mnist_pixels]
        generator_outfuns =[tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.tanh] 
        generator = MLP(lr=lr, scope="G", drop_out=1.0,
                             outfuns=generator_outfuns, 
                             layers=generator_layers)   
        generator_dropout = tf.placeholder(tf.float32, ())
        latent_sample = tf.placeholder(tf.float32, [num_samples, generator_layers[0]])
        generated_patterns = generator.update(latent_sample, generator_dropout)           
    #---------------------------------------------------------------------------
    #  Discriminator
    with tf.variable_scope('Discriminator'):    
        discriminator_layers = [n_mnist_pixels, 1024, 512, 256, 1]
        discriminator_outfuns =[tf.nn.relu, tf.nn.relu, tf.nn.relu,  tf.nn.sigmoid]
        discriminator = MLP(lr=lr, scope="D", drop_out=1.0,
                             outfuns=discriminator_outfuns,
                             layers=discriminator_layers)
        discriminator_dropout = tf.placeholder(tf.float32, ())
        data_sample = tf.placeholder(tf.float32, [num_samples, discriminator_layers[0]])
        D_probs = discriminator.update(data_sample, discriminator_dropout)
        G_probs = discriminator.update(generated_patterns, discriminator_dropout)
    #---------------------------------------------------------------------------
    # Losses
    D_loss = tf.reduce_mean(tf.log(D_probs + eps) + tf.log(1.0 - G_probs + eps))  
    G_loss = tf.reduce_mean(tf.log(G_probs + eps))
    #---------------------------------------------------------------------------   
    # Optimize
    D_train, Dw_change = discriminator.train(-D_loss)   
    G_train, Gw_change = generator.train(-G_loss)           
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
                curr_latent_sample = get_latent_sample()
                curr_data_sample = get_data_sample(iter)
                d_probs, d_loss, d_change, _ = session.run([D_probs, D_loss, Dw_change, D_train], 
                    feed_dict={latent_sample:curr_latent_sample,
                               data_sample:curr_data_sample, 
                               discriminator_dropout: dropout,
                               generator_dropout: 1.0})
                     
                # generator step 
                curr_latent_sample = get_latent_sample()
                g_probs, g_loss, g_change,_ = session.run([G_probs, G_loss, Gw_change, G_train], 
                    feed_dict={latent_sample:curr_latent_sample, 
                               discriminator_dropout: dropout,
                               generator_dropout: 1.0})        
                
                D_losses.append(d_loss)
                G_losses.append(g_loss)      
                D_changes.append(d_change)
                G_changes.append(g_change)
            
            HIST_D_losses.append(np.mean(D_losses))
            HIST_G_losses.append(np.mean(G_losses))
            HIST_D_changes.append(np.mean(D_changes))
            HIST_G_changes.append(np.mean(G_changes))  
                     
            patterns = session.run(generated_patterns, {latent_sample:fixed_latent_sample,
                                                        discriminator_dropout: dropout,
                                                        generator_dropout: 1.0})
            plotter.plot(
                HIST_G_losses, 
                HIST_D_losses,
                HIST_G_changes, 
                HIST_D_changes, patterns)
                
