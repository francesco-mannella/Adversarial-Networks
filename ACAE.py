# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from convolutional_mlp import MLP, linear, leaky_relu
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
"""
Adversarial autoencoder

* A multilayered perceptron as the autoencoder network 
    takes samples for the mnist dataset and reproduces them
"""
#---------------------------------------------------------------------------  
#---------------------------------------------------------------------------  
#---------------------------------------------------------------------------  
class AAE:

    eps = 1.0e-10

    def __init__(self, rlr, alr, weight_scale,
            encoder_outfuns, encoder_layers,
            decoder_outfuns, decoder_layers,
            adversarial_outfuns, adversarial_layers,
            encoder_dropouts=None, decoder_dropouts=None,
            adversarial_dropouts=None, encoder_bn=None,
            decoder_bn=None, adversarial_bn=None,
            bn_decay=None, encoder_convs=None,
            decoder_convs=None, adversarial_convs=None,
            encoder_deconvs=None, decoder_deconvs=None,
            adversarial_deconvs=None, encoder_strides=None,
            decoder_strides=None, adversarial_strides=None,
            encoder_copy_from=None, decoder_copy_from=None,
            adversarial_copy_from=None):

        self.rlr = rlr
        self.alr = alr
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        
        if bn_decay is None: bn_decay = 0.99

        self.encoder = MLP(scope="Encoder", 
                weight_scale=weight_scale,
                bn_decay=bn_decay,
                outfuns=encoder_outfuns, 
                layers_lens=encoder_layers,
                drop_out=encoder_dropouts,
                convs=encoder_convs, 
                deconvs=encoder_deconvs, 
                strides=encoder_strides, 
                batch_norms=encoder_bn)   

        self.decoder = MLP(scope="Decoder",
                weight_scale=weight_scale,
                bn_decay=bn_decay,
                outfuns=decoder_outfuns, 
                layers_lens=decoder_layers,        
                drop_out=decoder_dropouts,
                convs=decoder_convs, 
                deconvs=decoder_deconvs, 
                strides=decoder_strides, 
                batch_norms=decoder_bn)    

        self.adversarial = MLP(scope="Adversarial",
                weight_scale=weight_scale,
                bn_decay=bn_decay,
                outfuns=adversarial_outfuns,
                layers_lens=adversarial_layers,  
                drop_out=adversarial_dropouts,
                convs=adversarial_convs, 
                deconvs=adversarial_deconvs, 
                strides=adversarial_strides, 
                batch_norms=adversarial_bn)   

        self.make_graph()

    def make_graph(self):
    
        self.data_sample = tf.placeholder(tf.float32, 
                [None] + self.encoder_layers[0])
        self.prior_sample = tf.placeholder(tf.float32, 
                [None] + self.decoder_layers[0])
        self.drop_out = tf.placeholder(tf.float32, ())
        self.phase_train = tf.placeholder(tf.bool, ())
        
        self.hidden_patterns = self.encoder.update(self.data_sample,
                drop_out=self.drop_out, phase_train=self.phase_train)
        self.decoded_patterns = self.decoder.update(self.hidden_patterns, 
                drop_out=self.drop_out, phase_train=self.phase_train)   
        self.prior_generated_patterns = self.decoder.update(
                self.prior_sample, drop_out=self.drop_out,
                phase_train=self.phase_train)   
        
        self.D_probs = self.adversarial.update(self.prior_sample, 
                drop_out=self.drop_out, phase_train=self.phase_train)
        self.G_probs = self.adversarial.update(self.hidden_patterns, 
                drop_out=self.drop_out, phase_train=self.phase_train)        

        self.losses()
        self.optimize()

    def losses(self):

        self.R_loss = tf.losses.mean_squared_error(self.data_sample,
                self.decoded_patterns)
        self.D_loss = tf.reduce_mean(-tf.log(self.D_probs + self.eps)) \
                - tf.reduce_mean(tf.log(1.0 - self.G_probs + self.eps))  
        self.G_loss = tf.reduce_mean(-tf.log(self.G_probs + self.eps))
    
    def optimize(self):

        self.ER_train =  self.encoder.train(self.R_loss, lr=self.rlr)
        self.DR_train = self.decoder.train(self.R_loss, lr=self.rlr)
        self.D_train =  self.adversarial.train(self.D_loss, lr=self.alr)
        self.G_train =  self.encoder.train(self.G_loss, lr=self.alr)
    
    def train_step(self, session, curr_data_sample,
            curr_prior_sample, dropout=0.3):
        # reconstruction step -- (minimize reconstruction  error)
        #    data_sample -> encoder -> hidden_patterns 
        #                -> decoder -> decoded_patterns
        current_decoded_patterns, r_loss, _, _ = session.run(
                [self.decoded_patterns, self.R_loss, self.ER_train,
                    self.DR_train], feed_dict={ 
                        self.data_sample:curr_data_sample, 
                        self.drop_out: 1.0, self.phase_train: True})  

        # adversarial step -- (minimize discrimination error)
        #    data_sample -> encoder -> hidden_patterns 
        #                -> adversarial -> G_props
        #    prior_sample -> adversarial -> D_props
        d_loss, _ = session.run([self.D_loss, self.D_train], 
                feed_dict={self.data_sample:curr_data_sample, 
                    self.prior_sample:curr_prior_sample, 
                    self.drop_out: dropout, self.phase_train: True})               

        # adversarial step -- (maximize discrimination error)
        #    data_sample  -> encoder -> hidden_patterns 
        #                 -> adversarial -> G_props
        g_loss, _ = session.run([self.G_loss, self.G_train], 
                feed_dict={self.data_sample:curr_data_sample, 
                    self.drop_out: dropout, self.phase_train: True})                

        return current_decoded_patterns, r_loss, d_loss, g_loss

    def test_step(self, session, grid, test_data):
        # Current generation of image patterns in response to 
        #   the presentation of a 2D grid of values to the two 
        #   hidden units 
        grid_patterns = session.run(self.prior_generated_patterns, 
            feed_dict={self.prior_sample:grid, self.drop_out: 1.0, 
                self.phase_train: False})
        # Current activation of hidden units in response to the 
        #   presentation of data images
        curr_hidden_patterns, curr_decoded_patterns = session.run(
                [self.hidden_patterns, self.decoded_patterns], 
            feed_dict={self.data_sample:test_data, self.drop_out: 1.0, 
                self.phase_train: False})
        return grid_patterns, curr_hidden_patterns, curr_decoded_patterns
