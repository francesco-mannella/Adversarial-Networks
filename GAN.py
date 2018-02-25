# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from convolutional_mlp import MLP, linear

"""
Generative adversarial network

* A multilayered perceptron as the generator network - x_z = G(z)
    takes random samples for the latent z vector and gives a image x_z
   
* A multilayered perceptron as the discriminator network - p = D(x)
    takes x and gives a probability that x belongs 
    to the dataset D (x_d) and not to the generated distribution Z (x_z)

    the discriminator maximizes:
        D_loss = log(D(x_d)) + log(1 - D(G(z)) 
    the generator maximizes:
        G_loss = log( D(G(z)) 
"""

#-------------------------------------------------------------------------------

class GAN:

    eps = 1e-10

    def __init__(self, learning_rate,
            generator_layers, generator_dropouts, generator_outfuns, 
            discriminator_layers, discriminator_dropouts, discriminator_outfuns):
        
        self.generator_layers = generator_layers
        self.discriminator_layers = discriminator_layers
        self.generator_dropouts = generator_dropouts
        self.discriminator_dropouts = discriminator_dropouts
        
        self.generator = MLP(scope="Generator", 
                lr=learning_rate,
                drop_out=generator_dropouts,
                outfuns=generator_outfuns, 
                layers_lens=generator_layers) 

        self.discriminator = MLP(scope="Discriminator", 
                        lr=learning_rate, 
                        drop_out=discriminator_dropouts,
                        outfuns=discriminator_outfuns,
                        layers_lens=discriminator_layers)

        self.make_graph()

    def make_graph(self):
        
        # spread graph
        # placeholders
        self.latent_sample = tf.placeholder(tf.float32, (None, self.generator_layers[0]))
        self.data_sample = tf.placeholder(tf.float32, (None, self.discriminator_layers[0]))
        self.curr_generator_dropout = tf.placeholder(tf.float32, ())
        self.curr_discriminator_dropout = tf.placeholder(tf.float32, ())
        # graph branches
        self.generated_patterns = self.generator.update(self.latent_sample, drop_out=self.curr_generator_dropout)           
        self.D_probs = self.discriminator.update(self.data_sample, drop_out=self.curr_discriminator_dropout)
        self.G_probs = self.discriminator.update(self.generated_patterns, drop_out=self.curr_discriminator_dropout)
    
        # train graph
        self.losses()
        
        self. D_train, self.Dw_change = self.discriminator.train(-self.D_loss)   
        self.G_train, self.Gw_change = self.generator.train(-self.G_loss)      
    
    def losses(self):

        self.D_loss = tf.reduce_mean(tf.log(self.D_probs + self.eps) + tf.log(1.0 - self.G_probs + self.eps))  
        self.G_loss = tf.reduce_mean(tf.log(self.G_probs + self.eps))

    def train_step(self, session, curr_discr_latent_sample, 
            curr_gen_latent_sample, curr_data_sample,
            generator_droprate=1.0, discriminator_droprate=0.3):

        # discriminator step
        d_probs, d_loss, d_change, _ = session.run([self.D_probs,
            self.D_loss, self.Dw_change, self.D_train], feed_dict={
                self.latent_sample:curr_discr_latent_sample, 
                self.data_sample:curr_data_sample, 
                self.curr_discriminator_dropout: discriminator_droprate, 
                    self.curr_generator_dropout: generator_droprate})

        # generator step 
        g_probs, g_loss, g_change,_ = session.run([self.G_probs, 
            self.G_loss, self.Gw_change, self.G_train], feed_dict={
                self.latent_sample:curr_gen_latent_sample, 
                self.curr_discriminator_dropout: discriminator_droprate,
                self.curr_generator_dropout: generator_droprate})  

        return  d_probs, d_loss, d_change, g_probs, g_loss, g_change

    def generative_test_step(self, session, curr_latent_sample):

        patterns = session.run(self.generated_patterns, feed_dict={
            self.latent_sample:curr_latent_sample,
            self.curr_discriminator_dropout: 1.0, 
            self.curr_generator_dropout: 1.0})

        return patterns


