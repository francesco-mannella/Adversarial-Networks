# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from convolutional_mlp import MLP, linear, leaky_relu
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
"""
Adversarial autoautoencoder

* A multilayered perceptron as the autoautoencoder network 
    takes samples for the mnist dataset and reproduces them
"""
#---------------------------------------------------------------------------  
#---------------------------------------------------------------------------  
#---------------------------------------------------------------------------  
class CAE:

    def __init__(self, rlr, weight_scale, autoencoder_outfuns,
            autoencoder_layers, autoencoder_dropouts=None,
            autoencoder_bn=None, bn_decay=None,
            autoencoder_convs=None, autoencoder_deconvs=None,
            autoencoder_strides=None, autoencoder_copy_from=None):

        self.rlr = rlr
        self.autoencoder_layers = autoencoder_layers
        self.autoencoder_layers = autoencoder_layers
        
        if bn_decay is None: bn_decay = 0.99

        self.autoencoder = MLP(scope="autoencoder", 
                weight_scale=weight_scale,
                bn_decay=bn_decay,
                outfuns=autoencoder_outfuns, 
                layers_lens=autoencoder_layers,
                drop_out=autoencoder_dropouts,
                convs=autoencoder_convs, 
                deconvs=autoencoder_deconvs, 
                strides=autoencoder_strides, 
                batch_norms=autoencoder_bn)   

        self.make_graph()

    def make_graph(self):
    
        self.data_sample = tf.placeholder(tf.float32, 
                [None] + self.autoencoder_layers[0])
        self.drop_out = tf.placeholder(tf.float32, ())
        self.phase_train = tf.placeholder(tf.bool, ())
        
        self.decoded_patterns = self.autoencoder.update(self.data_sample, 
                drop_out=self.drop_out, phase_train=self.phase_train)   
        
        self.losses()
        self.optimize()

    def losses(self):

        self.R_loss = tf.losses.mean_squared_error(self.data_sample,
                self.decoded_patterns)
    
    def optimize(self):

        self.AER_train =  self.autoencoder.train(self.R_loss, lr=self.rlr)
    
    def train_step(self, session, curr_data_sample, dropout=0.3):
        # reconstruction step -- (minimize reconstruction  error)
        #    data_sample -> autoencoder -> hidden_patterns 
        #                -> autoencoder -> decoded_patterns
        current_decoded_patterns, r_loss, _ = session.run(
                [self.decoded_patterns, self.R_loss, self.AER_train],
                feed_dict={ self.data_sample:curr_data_sample,
                    self.drop_out: 1.0, self.phase_train: True})  

        return current_decoded_patterns, r_loss 

    def test_step(self, session, test_data):
        # Current decoding in response to the 
        #   presentation of data images
        curr_decoded_patterns = session.run( self.decoded_patterns,
                feed_dict={self.data_sample:test_data, self.drop_out:
                    1.0, self.phase_train: False})
        return curr_decoded_patterns
