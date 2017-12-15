import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.set_printoptions(suppress=True,
                    precision=5,
                    linewidth=10000)

def add_bias(tensor):   
    ndim = len(tensor.get_shape())
    ldim = tensor.get_shape()[:-1].as_list()
    bias = np.ones(ldim+[1])
    return tf.concat([tensor, bias], 1)

class BackProp(object):

    def __init__(self, num_units_per_layer=[2, 2, 1], 
            learning_rate=0.1, activation=tf.nn.relu, 
            rng = None, scope="bp"):
        """
        :param num_units_per_layer: List of integers. 
            The number of units for each layer, from 
            input to output
        :param learning_rate: learning rate
        :param activation: activation function for network units
        :param rng: random number generator
        """
    
        with tf.variable_scope(scope):

            self.num_layers = len(num_units_per_layer)
            self.num_units_per_layer = num_units_per_layer
            self.learning_rate = learning_rate
            self.activation = activation
    
            # random generator
            self.rng = rng
            if rng is None: self.rng = np.random.RandomState()
    
            # units 
            self.units = [None for x in range(self.num_layers)]
    
            # weights 
            self.weights = []
            bias = 1
            self.id = id(self)
            for weight in range(self.num_layers-1):
                n_input = self.num_units_per_layer[weight] + bias 
                n_output = self.num_units_per_layer[weight + 1]
                #r = np.sqrt(6.0/(n_input + n_output + 1)) 
                #w_init = self.rng.uniform(-r, r, [n_input, n_output])
                w_init = self.rng.randn(n_input, n_output)
                current_weight_matrix = tf.get_variable('w{:03d}'.format(weight),
                        initializer = tf.constant(w_init, dtype="float32"), 
                        trainable = True)  
                self.weights.append(current_weight_matrix)
                
            self.loss = None

    def iteration(self, pattern):
        """
        :param pattern: vector of float, input pattern
        """
        self.units[0] = pattern
        for layer in range(self.num_layers - 1):
            biased = add_bias(self.units[layer])
            self.units[layer + 1] = self.activation(
                    tf.matmul(biased, self.weights[layer]))

    def update(self, target):
        loss = self.loss()
        train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        return loss, train
    
