# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# utils
def linear(x): 
    """  Linear activation """
    return x

def leaky_relu(x, alpha=0.2): 
    """  Leaky relu 
        Args:
            x:        A Tensor representing preactivation values.
            alpha:    Slope of the activation function at x < 0.
        Returns:
            The activation value.
    """
    return tf.nn.relu(x) -alpha*tf.nn.relu(-x)

def tf_shape(x):
    """
    compute the shape of a tensor as a list
    Args:
        x:        A Tensor
    """
    return x.get_shape().as_list()

def tf_reshape(x, shape):
    if shape[0] is None:
        x = tf.reshape(x, [-1] + shape[1:])
    else:
        x = tf.reshape(x, shape)
    return x


def flatten_shape(shape):
    """
    Flatten a 4D BHWD shape to a BL shape
    Args:
        shape: list, the dimensions of a tensor
    Returns:
        the flattened shape
    """ 
    if len(shape) == 4:
        return [shape[0], np.prod(shape[1:])]
    return shape

def flatten(x):
    """
    Flatten a 4D BHWD tensor to a BL tensor
    Args:
        x: a tensor
    Returns:
        the flattened tensor
    """ 
    shape = tf_shape(x)
    x = tf_reshape(x,flatten_shape(shape))
    return x

def unflatten_shape(shape, layers=1):
    """
    Unflatten a 2D BL shape to a BHWD shape
    Args:
        shape: list, the dimensions of a tensor
        layers: int, the number of depth layers
    Returns:
        the unflattened BHWD shape
    """ 
    if len(shape) == 2:
        side = shape[1]/layers
        side = int(np.sqrt(side))
        return  [shape[0], side, side, layers]
    return shape

def unflatten(x, layers=1):
    """
    Unflatten a 2D BL tensor to a BHWD tensor
    Args:
        x: a tensor
        layers: int, the depth dimension
    Returns:
        the unflattened BHWD tensor
    """ 
    shape = tf_shape(x)
    x = tf_reshape(x, unflatten_shape(shape, layers))
    return x

def get_outshape(inp_shape, w_shape, k, conv=True):
    """
    Compute the output shape of a convolution/deconvolution given 
    a input layer and a weight tensor
    Args:
        inp_shape:    list, the shape of the input tensor
        w_shape:      list, the shape of the weight tensor
        k:            int, the value of strides ([1, k, k, 1])
        conv:         bool, it is a convolution (True) or a deconvolution (False)
        Returns:
            a list defining the output shape
    """
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
    """
    Convolution. Control for shapes
    Args:
        x:            the input tensor
        W:            the weight tensor
        k:            int, the value of strides ([1, k, k, 1])  
        inp_layers:   int, the depth dimension of the input layer 
    Returns:
        a tensor
    """ 
    x = unflatten(x, inp_layers)
    strides=[1, k, k, 1]   
    return tf.nn.conv2d(x, W, strides=strides, padding='SAME')   

def deconv2d(x, W, k=1, inp_layers=1):  
    """
    Deonvolution. Control for shapes
    Args:
        x:            the input tensor
        W:            the weight tensor
        k:            int, the value of strides ([1, k, k, 1])  
        inp_layers:   int, the depth dimension of the input layer 
    Returns:
        a tensor
    """ 
    x = unflatten(x, inp_layers)
    out_shape = get_outshape(tf_shape(x), tf_shape(W), k, conv=False)  
    strides=[1, k, k, 1]       
    return tf.nn.conv2d_transpose(x, W, output_shape=out_shape,
                                   strides=strides, padding='SAME')

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def batch_norm(x, n_out, decay=0.99, scope="layer", epsilon=0.001, phase_train=False):
    """
    Batch normalization on raw layers or convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 2D BL or 4D BHWD input maps
        n_out:       integer, depth of input maps
        decay:       Float, the decay rate of the exp. moving avg
        phase_train: tf.bool, true indicates training phase
    Return:
        normed:      batch-normalized maps
    """
    shape = tf_shape(x)
    batch_mean, batch_var = tf.nn.moments(x, range(len(shape)-1))
    
    with tf.variable_scope(scope):

        with tf.variable_scope("bn"):
            
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                         name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                          name='gamma', trainable=True)
            
            batch_mean, batch_var = tf.nn.moments(x, range(len(shape)-1), name='moments')
            
            ema = tf.train.ExponentialMovingAverage(decay=decay)
    
            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
    
            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)
            
    return normed
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class MLP(object):
    '''
    Multilayer perceptron
    '''
    def __init__(self, lr, outfuns, layers_lens,  convs=None, deconvs=None, 
                 batch_norms=None, bn_decay=0.999, strides=None, copy_from=None, 
                 drop_out=None, weight_scale=0.02, scope="bp"): 
        """
        Args:
             lr:            float, learning rate 
             outfuns:       list(callable), list of activation functions for each layer
             layers_lens:   list(int), number of units per layer
             convs:         list(array), Kernel weight arrays. If None the layer is not convolutional
             deconvs:       list(array), Kernel weight arrays. If None the layer is not deconvolutional
             batch_norms:   list(bool), bools. If True the layer has batch normalization
             bn_decay:      float, decay of exp. mov. avg for the batch normalization
             copy_from:     list(int), indices of layers from which to copy weights
             strides:       list(int), strides for each conv/deconv. a value 'k' means [1, k, k, 1]
             drop_out:      list(bool), which layer has dropout
             weight_scale:  the initial scaling of weights
             scope:         string, a label defining the scope of the MLP object's variables
        """
        self.lr = lr
        self.layers_lens = layers_lens
        self.strides = strides
        self.outfuns = outfuns
        self.copy_from = copy_from
        self.convs = convs
        self.deconvs = deconvs
        self.batch_norms = batch_norms
        self.scope = scope
        self.drop_out = drop_out
        self.bn_decay = bn_decay
        
        # control not given parameters 
        self.layers_lens = [ x if not np.isscalar(x) else [x] for x in self.layers_lens ]
        if self.strides is None :
            self.strides = [ None for x in range(len(layers_lens) - 1)]   
        if self.convs is None :
            self.convs = [ None for x in range(len(layers_lens) - 1)]        
        if self.deconvs is None :
            self.deconvs = [ None for x in range(len(layers_lens) - 1)]
        if self.batch_norms is None :
            self.batch_norms = [ False for x in range(len(layers_lens) - 1)]
        if self.copy_from is None :
            self.copy_from = [ None for x in range(len(layers_lens) - 1)]
        if self.drop_out is None :
            self.drop_out = [ False for x in range(len(layers_lens) - 1)]
        
        # lists of network variables  
        self.weights = []
        self.biases = []
        self.layers = []
        self.shapes = []
        
        # network building graph
        with tf.variable_scope(scope):
   
            for curr_l0, (inp_shape_, out_shape_,  
                          has_batch_norm, conv, 
                          deconv, copy_index, outfun) \
                    in enumerate(zip(self.layers_lens[:-1], self.layers_lens[1:], 
                        self.batch_norms, self.convs, 
                        self.deconvs, self.copy_from, self.outfuns)):
                
                # layer building graph
                with tf.variable_scope("layer-{:03d}".format(curr_l0)):

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
            
                    # weight and bias initializers
                    w_initial = tf.truncated_normal(w_shape, stddev=weight_scale)
                    b_initial = tf.constant(0.0, shape=b_shape)
                    
                    # weights
                    if copy_index is not None:
                        assert curr_l0 > copy_index, "must copy from previous layer"
                        weight = self.weights[copy_index]
                    elif copy_index is None:
                        weight = tf.get_variable(
                            name="w",
                            dtype=tf.float32,
                            initializer=w_initial) 
        
                    # biases
                    bias = None
                    if has_batch_norm == False:
                        bias = tf.get_variable(
                            name="b", 
                            dtype=tf.float32, 
                            initializer=b_initial)
                 
                    # store variables
                    self.weights.append(weight)
                    self.biases.append(bias)
       
    def update(self, inp, drop_out=None, phase_train=True): 
        """
        Spreading of activations throughout the layers of the MLP objects
        Args:
            inp:         tf.tensor, the current input pattern
            drop_out:    tensor, probability of retain units
            phase_train: tf.bool, indicates if within the training phase 
        Returns:
            output_layer: tf.tensor
        """

        if drop_out is None: drop_out = tf.constant(1.0)

        num_layers = len(self.weights)
        self.layers = []
        
        # network spreading graph
        with tf.variable_scope(self.scope):

            self.layers.append(inp)
            for curr_l0, (weight, bias, has_batch_norm,
                          conv, deconv, copy_index, strides, 
                          outfun) \
                in enumerate(zip(self.weights, self.biases, self.batch_norms, 
                                 self.convs, self.deconvs, self.copy_from, self.strides, 
                                 self.outfuns)):
                
                
                
                scope = "layer-{:03d}".format(curr_l0)
                print "{}_{} weights     {}".format(self.scope, curr_l0, tf_shape(weight))

                # layer spreading graph
                with tf.variable_scope(scope):
                    
                    if copy_index is not None:
                        print "{}_{} copy        YES".format(self.scope, curr_l0)
                    else:
                        print "{}_{} copy        NO".format(self.scope, curr_l0)

                    # weighted sum (conv2d, deconv2d and flat cases)
                    input_layer = self.layers[curr_l0] 
                    if conv is not None:           # convolution --------
                        print "{}_{} conv        YES".format(self.scope, curr_l0)
                        print "{}_{} deconv      NO".format(self.scope, curr_l0)
                        print "{}_{} dense       NO".format(self.scope, curr_l0)
                        output_layer = conv2d(
                            input_layer, weight, strides, conv[-2]) 
                    elif deconv is not None:       # deconvolution ------
                        print "{}_{} conv        NO".format(self.scope, curr_l0)
                        print "{}_{} deconv      YES".format(self.scope, curr_l0)
                        print "{}_{} dense       NO".format(self.scope, curr_l0)
                        output_layer = deconv2d(
                            input_layer, weight,strides, deconv[-1]) 
                    else:    
                        print "{}_{} conv        NO".format(self.scope, curr_l0)
                        print "{}_{} deconv      NO".format(self.scope, curr_l0)
                        print "{}_{} dense       YES".format(self.scope, curr_l0)                      # dense --------------
                        input_layer = flatten(input_layer)
                        if copy_index is not None:
                            output_layer = tf.matmul(
                                input_layer, tf.transpose(weight))
                        else:
                            output_layer = tf.matmul(input_layer, weight)
                            
                    if has_batch_norm == True:
                        # batch norm
                        print "{}_{} batch_norm  YES".format(self.scope, curr_l0)
                        output_layer = batch_norm(
                            output_layer, 
                            n_out=tf_shape(output_layer)[-1],
                            decay=self.bn_decay,
                            scope=scope, phase_train=phase_train)
                    elif has_batch_norm == False:   
                        # only bias  
                        print "{}_{} batch_norm  NO".format(self.scope, curr_l0)
                        output_layer += bias

                    # transfer function
                    output_layer = outfun(output_layer)
                    
                    # dropout - avoids drop_out of the last layer 
                    if self.drop_out[curr_l0] == True: 
                        print "{}_{} dropout     YES".format(self.scope, curr_l0)
                        output_layer = tf.nn.dropout(output_layer, drop_out) 
                    else:
                        print "{}_{} dropout     NO".format(self.scope, curr_l0)
 
                    print
                    
                    # store layer
                    self.layers.append(output_layer)
                                
        return self.layers[-1]
      
    def train(self, loss, lr = None, optimizer = None):
        """
        Gradient optimization
        Args:
            loss: tf.tensor, the value to be optimized
            optimizer: default is tf.train.AdamOptimizer
        Return:
            apply_gradients_op: the pointer to the minimizing operation and 
            norm: float, the mean norm of the current gradient tensors
        """
        # control not given parameters 
        if lr is None: 
            lr = self.lr
        if optimizer is None: 
            optimizer = tf.train.AdamOptimizer
       
        # build optimizer 
        opt = optimizer(lr)
        # collect all trainiable variables for this object
        trainables = tf.trainable_variables()
        train_vars = [var for var in trainables if self.scope in var.name]
        # compute gradients and train
        grads_and_vars = opt.compute_gradients(loss, var_list=train_vars)   
        norm = tf.reduce_mean([tf.norm(grad) for grad,_ in  grads_and_vars if grad is not None])

        return opt.apply_gradients(grads_and_vars), norm

