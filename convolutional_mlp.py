# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#-------------------------------------------------------------------------------
# utils
def linear(x): return x

def tf_shape(x):
    return x.get_shape().as_list()

def flatten_shape(shape):
    if len(shape) == 4:
        return [shape[0], np.prod(shape[1:])]
    return shape

def flatten(x):
    shape = tf_shape(x)
    x = tf.reshape(x, flatten_shape(shape))
    return x

def unflatten_shape(shape, layers=1):
    if len(shape) == 2:
        side = shape[1]/layers
        side = int(np.sqrt(side))
        return  [shape[0], side, side, layers]
    return shape

def unflatten(x, layers=1):
    shape = tf_shape(x)
    x = tf.reshape(x, unflatten_shape(shape, layers))
    return x

def get_outshape(inp_shape, w_shape, k, conv=True):
    inp_shape = unflatten_shape(inp_shape)
    out_shape = inp_shape[:]
    if conv==True:
        out_shape[1] /= k
        out_shape[2] /= k
        out_shape[3] = w_shape[-1]  
    elif conv==False:
        out_shape[1] *= k
        out_shape[2] *= k
        out_shape[3] = w_shape[-2]         
    return out_shape

def conv2d(x, W, k=1, inp_layers=1):  
    x = unflatten(x, inp_layers)
    strides=[1, k, k, 1]   
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')   

def deconv2d(x, W, k=1, inp_layers=1):  
    x = unflatten(x, inp_layers)
    out_shape = get_outshape(tf_shape(x), tf_shape(W), k, conv=False)  
    strides=[1, k, k, 1]       
    return tf.nn.conv2d_transpose(x, W, output_shape=out_shape,
                                   strides=strides, padding='SAME')

def max_pool(x, k = 2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1],
                          strides=[1, k, k, 1], padding='SAME')

#-------------------------------------------------------------------------------
class MLP(object):
    '''
    Multilayer perceptron
    '''
    def __init__(self, lr, outfuns, layers_lens, convs, deconvs, 
            strides, copy_from=None, drop_out=1.0, scope="bp"): 
        """
        :param lr: float, learning rate 
        :param outfuns: list(callable), list of activation functions for each layer
        :param layers_lens: list(int), number of units per layer
        :param convs: list(array), Kernel weight arrays. If None the layer is not convolutional
        :param deconvs: list(array), Kernel weight arrays. If None the layer is not deconvolutional
        :param copy_from: list(int), indices of layers from which to copy weights
        :param strides: list(int), strides for each conv/deconv. a value 'k' means [1, k, k, 1]
        :param drop_out: float, probability of retain units
        :param scope: string, a label defining the scope of the MLP object's variables
        """
        self.lr = lr
        self.layers_lens = layers_lens
        self.convs = convs
        self.deconvs = deconvs
        self.strides = strides
        self.outfuns = outfuns
        self.copy_from = copy_from
        if self.copy_from is None :
            self.copy_from = [ None for x in range(len(layers_lens) - 1)]
        self.drop_out = drop_out

        # lists of network variables  
        self.weights = []
        self.biases = []
        self.layers = []
        self.shapes = []
    
        # compute first layer shape
        for curr_l0, (inp_shape_, out_shape_, conv, deconv, copy_index, outfun) \
                in enumerate(zip(layers_lens[:-1], layers_lens[1:], 
                    self.convs, self.deconvs, self.copy_from, self.outfuns)):

            inp_shape = [None] + inp_shape_[:]
            out_shape = [None] + out_shape_[:] 
            
            # control lower-upper-layer shapes
            strides = self.strides[curr_l0]
            if conv is not None:
                inp_shape = unflatten_shape(inp_shape, conv[-2])
                out_shape = unflatten_shape(out_shape, conv[-1])
            elif deconv is not None:
                inp_shape = unflatten_shape(inp_shape, deconv[-1])
                out_shape = unflatten_shape(out_shape, deconv[-2])
            else:   
                inp_shape = flatten_shape(inp_shape)
                out_shape = flatten_shape(out_shape)
            self.shapes.append([inp_shape, out_shape])

            # Shapes of weights and biases
            if conv is not None:
                w_shape = conv
                b_shape = [1, conv[-1]]
            elif deconv is not None:
                w_shape = deconv
                b_shape = [1, deconv[-2]]
            else:
                w_shape = [inp_shape[1], out_shape[1]]   
                b_shape = [1, out_shape[1]]
    
            # build weight and bias tensors
            scale = np.sqrt(6.0/ ( np.prod(inp_shape[1:]) + np.prod(out_shape[1:])) )
            w_initial = tf.truncated_normal(w_shape, stddev=scale)
            b_initial = tf.constant(0.0, shape=b_shape)
            
            if copy_index is not None:
                assert curr_l0 > copy_index, "must copy from previous layer"
                weight = self.weights[copy_index]
            elif copy_index is None:
                weight = tf.get_variable(
                    name="w{}_{}".format(scope, curr_l0),
                    dtype=tf.float32,
                    initializer=w_initial)

            bias = tf.get_variable(
                name="b{}_{}".format(scope, curr_l0), 
                dtype=tf.float32, 
                initializer=b_initial)
            
            # store in the network lists
            self.weights.append(weight)
            self.biases.append(bias)
         
        # a unique list with all the tensors needed for training of this object    
        uncopied = [x for x in range(len(self.copy_from)) if self.copy_from[x] is None]
        self.weights_biases = [self.weights[x]  for  x in uncopied] + [self.biases]
    
    def update(self, inp, drop_out=None): 
        """
        Spreading of activations throughout the layers of the MLP objects
        :param inp: tf.tensor, the current input pattern
        :param drop_out: tensor, probability of retain units
        :return: tf.tensor, the output layer
        """
        # set drop_out probability
        if drop_out is None: drop_out = self.drop_out
        
        num_layers = len(self.layers_lens)
        self.layers = []
        
        # spreading graph
        self.layers.append(inp)
        for curr_l0, (weight, bias, conv, deconv, copy_index, strides, outfun) \
            in enumerate(zip(self.weights, self.biases, self.convs, 
                             self.deconvs, self.copy_from, self.strides, self.outfuns)):
            
            # weighted sum (conv2d, deconv2d and flat cases)
            input_layer = self.layers[curr_l0] 
            if conv is not None:
                output_layer = conv2d(input_layer, weight, strides, conv[-2]) 
            elif deconv is not None:
                output_layer = deconv2d(input_layer, weight, strides, deconv[-1]) 
            else:
                input_layer = flatten(input_layer)
                if copy_index is not None:
                    output_layer = tf.matmul(input_layer, tf.transpose(weight))
                else:
                    output_layer = tf.matmul(input_layer, weight)

            # transfer function
            output_layer = outfun(output_layer + bias)
              
            # avoid drop_out of the last layer 
            if curr_l0 < num_layers - 1 and self.drop_out < 1.0 and conv is None and deconv is None: 
                output_layer = tf.nn.dropout(output_layer, drop_out) 
            self.layers.append(output_layer)
        
        return output_layer
      
    def train(self, loss, lr = None, optimizer = None):
        """
        Gradient optimization
        :param loss: tf.tensor, the value to be optimized
        :param optimizer: default is tf.train.AdamOptimizer
        :return: (optimizing operation, float), the pointer to the minimizing operation and 
                the mean norm of the current gradient tensors
        """
        if lr is None: lr = self.lr
        if optimizer is None: optimizer = tf.train.AdamOptimizer
        opt = optimizer(lr)
        grads_and_vars = opt.compute_gradients(loss, var_list=self.weights_biases)   
        norm = tf.reduce_mean([tf.norm(g) for g,_ in  grads_and_vars])
        return opt.apply_gradients(grads_and_vars), norm

