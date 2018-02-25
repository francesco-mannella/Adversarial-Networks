# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from convolutional_mlp import MLP, linear, leaky_relu
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
"""
Adversarial autoencoder

* A multilayered perceptron as the autoencoder network 
    takes samples for the mnist dataset and reproduces them
    
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
#---------------------------------------------------------------------------  
#---------------------------------------------------------------------------  
#---------------------------------------------------------------------------  

class AAE:

    eps = 1.0e-10

    def __init__(self, 
            rlr, alr, weight_scale,
            encoder_outfuns, encoder_layers,
            decoder_outfuns, decoder_layers,
            adversarial_outfuns, adversarial_layers):
        
        self.rlr = rlr
        self.alr = alr
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        self.encoder = MLP(scope="self.Encoder", 
                weight_scale=weight_scale,
                outfuns=encoder_outfuns, 
                layers_lens=encoder_layers)   

        self.decoder = MLP(scope="self.Decoder",
                weight_scale=weight_scale,
                outfuns=decoder_outfuns, 
                layers_lens=decoder_layers)    

        self.adversarial = MLP(scope="self.Adversarial",
                weight_scale=weight_scale,
                outfuns=adversarial_outfuns,
                layers_lens=adversarial_layers)   

        self.make_graph()

    def make_graph(self):
    
        self.data_sample = tf.placeholder(tf.float32, [None, self.encoder_layers[0]])
        self.prior_sample = tf.placeholder(tf.float32, [None, self.decoder_layers[0]])
        self.drop_out = tf.placeholder(tf.float32, ())
        self.phase_train = tf.placeholder(tf.bool, ())
        
        self.hidden_patterns = self.encoder.update(self.data_sample,
                drop_out=self.drop_out, phase_train=self.phase_train)
        self.decoded_patterns = self.decoder.update(self.hidden_patterns, 
                drop_out=self.drop_out, phase_train=self.phase_train)   
        self.prior_generated_patterns = self.decoder.update(self.prior_sample, 
                drop_out=self.drop_out, phase_train=self.phase_train)   
        
        self.D_probs = self.adversarial.update(self.prior_sample, 
                drop_out=self.drop_out, phase_train=self.phase_train)
        self.G_probs = self.adversarial.update(self.hidden_patterns, 
                drop_out=self.drop_out, phase_train=self.phase_train)        

        self.losses()
        self.optimize()


    def losses(self):

        self.R_loss = tf.losses.mean_squared_error(self.data_sample, self.decoded_patterns)
        self.D_loss = tf.reduce_mean(-tf.log(self.D_probs + self.eps) - tf.log(1.0 - self.G_probs + self.eps))  
        self.G_loss = tf.reduce_mean(-tf.log(self.G_probs + self.eps))
    
    def optimize(self):

        self.ER_train =  self.encoder.train(self.R_loss, lr=self.rlr)
        self.DR_train = self.decoder.train(self.R_loss, lr=self.rlr)
        self.D_train =  self.adversarial.train(self.D_loss, lr=self.alr)
        self.G_train =  self.encoder.train(self.G_loss, lr=self.alr)
    
    def train_step(self, session, curr_data_sample, curr_prior_sample, dropout=0.3):
        # reconstruction step -- (minimize reconstruction  error)
        #    data_sample -> encoder -> hidden_patterns -> decoder -> decoded_patterns
        current_decoded_patterns, r_loss, _, _ = session.run(
                [self.decoded_patterns, self.R_loss, self.ER_train, self.DR_train], 
                feed_dict={self.data_sample:curr_data_sample, 
                    self.drop_out: 1.0, self.phase_train: True})  

        # adversarial step -- (minimize discrimination error)
        #    data_sample  -> encoder -> hidden_patterns -> adversarial -> G_props
        #    prior_sample                               -> adversarial -> D_props
        d_loss, _ = session.run([self.D_loss, self.D_train], 
                feed_dict={self.data_sample:curr_data_sample, 
                    self.prior_sample:curr_prior_sample, 
                    self.drop_out: dropout, self.phase_train: True})               

        # adversarial step -- (maximize discrimination error)
        #    data_sample  -> encoder -> hidden_patterns -> adversarial -> G_props
        g_loss, _ = session.run([self.G_loss, self.G_train], 
                feed_dict={self.data_sample:curr_data_sample, 
                    self.drop_out: dropout, self.phase_train: True})                

        return current_decoded_patterns, r_loss, d_loss, g_loss

    def test_step(self, session, grid, test_data):
        # Current generation of image patterns in response to 
        #   the presentation of a 2D grid of values to the two hidden units 
        grid_patterns = session.run(self.prior_generated_patterns, 
            feed_dict={self.prior_sample:grid, self.drop_out: 1.0, 
                self.phase_train: False})
        # Current activation of hidden units in response to the presentation of data images
        curr_hidden_patterns = session.run(self.hidden_patterns, 
            feed_dict={self.data_sample:test_data, self.drop_out: 1.0, 
                self.phase_train: False})

        return grid_patterns, curr_hidden_patterns

