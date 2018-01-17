# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#-------------------------------------------------------------------------------
# utils
def linear(x): return x
#-------------------------------------------------------------------------------
class MLP(object):
    '''
    Multilayer perceptron
    '''
    def __init__(self, lr, outfuns, layers, drop_out=1.0, scope="bp"): 
        """
        :param lr: float, learning rate 
        :param outfuns: list(callable), list of activation functions for each layer
        :param layers: list(int), number of units per layer
        :param drop_out: float, probability of retain units
        :param scope: string, a label defining the scope of the MLP object's variables
        """
        self.lr = lr
        self.outfuns = outfuns
        self.drop_out = drop_out
        self.scope = scope
    
        self.weights = []
        self.biases = []
        self.shapes = []
        for i in range(len(layers) - 1):
  
            shape = (layers[i], layers[i + 1])
            scale = 3*np.sqrt(6.0/ ( shape[0] + shape[1]) )
            scale = 0.1
            w_init = tf.random_uniform_initializer(-scale, scale)
            b_init = tf.constant_initializer(0.)       
            weight = tf.get_variable(
                name="w{}_{}".format(scope, i), 
                dtype=tf.float32, 
                shape=shape,
                initializer=w_init)    
            bias = tf.get_variable(
                name="b{}_{}".format(scope, i), 
                dtype=tf.float32, 
                shape=[1, shape[1]],
                initializer=b_init)
            self.weights.append(weight)
            self.biases.append(bias)
            self.shapes.append(shape)
            
        self.weights_biases = self.weights + self.biases

    def update(self, inp, drop_out=None): 
        """
        Spreading of activations throughout the layers of the MLP objects
        :param inp: tf.tensor, the current input pattern
        :param drop_out: tensor, probability of retain units
        :return: tf.tensor, the output layer
        """
        if drop_out is None: drop_out = self.drop_out
        n = len(self.weights)
        self.layers = []
        self.layers.append(inp)
        for i, (w, b, f) in enumerate(zip(self.weights, self.biases, self.outfuns)):
            h = f(tf.matmul(self.layers[i], w) + b)
            if i < n - 1: 
                h = tf.nn.dropout(h, drop_out) 
            self.layers.append(h)
        return h
      
    def train(self, loss, lr = None):
        """
        Gradient optimization
        :param loss: tf.tensor, the value to be optimized
        :return: (optimizing operation, float), the pointer to the minimizing operation and 
                the mean norm of the current gradient tensors
        """
        if lr is None: lr = self.lr
        opt = tf.train.AdamOptimizer(lr)
        
        grads_and_vars = opt.compute_gradients(loss, var_list=self.weights_biases)   
        norm = tf.reduce_mean([tf.norm(g) for g,_ in  grads_and_vars])
        return opt.apply_gradients(grads_and_vars), norm
#---------------------------------------------------------------------------
