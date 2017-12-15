import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from backprop import BackProp

#-------------------------------------------------------------------------------
# only current needed GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#-------------------------------------------------------------------------------
# set the seed for random numbers generation
current_seed = np.fromstring(os.urandom(4), dtype=np.uint32)[0]
print "seed:%d" % current_seed
rng = np.random.RandomState(current_seed) 
#-------------------------------------------------------------------------------

graph = tf.Graph()
with graph.as_default():

    # mnist

    # import the mnist class
    from mnist import MNIST

    # init with the 'data' dir
    mndata = MNIST('./data')

    # Load data
    mndata.load_training()
    mndata.load_testing()

    # The number of pixels per side of all images
    img_side = 28

    # Each input is a raw vector.
    # The number of units of the network
    # corresponds to the number of input elements
    n_mnist_pixels = img_side * img_side

    # lengths of datatasets patterns
    n_train = len(mndata.train_images)
    n_test = len(mndata.test_images)

    # samples to generate per iteration
    num_samples = 20

    # training iterations
    iterations = 100
    
    # generator network
    num_latent_variables = 10
    generator_learning_rate = 0.01
    generator = bp = BackProp(activation=tf.nn.relu, 
                num_units_per_layer=[num_latent_variables, 20, 40, n_mnist_pixels],
                learning_rate=generator_learning_rate, rng=rng, scope="generator")
    
    # discriminator network
    discriminator_learning_rate = 0.01
    discriminator = bp = BackProp(activation=tf.nn.relu, 
                num_units_per_layer=[n_mnist_pixels, 40, 20, 1],
                learning_rate=discriminator_learning_rate, rng=rng, scope="discriminator") 
    
    data = tf.constant(np.array(mndata.train_images)/255.0, dtype="float32")
    
    # iteration graph
    
    # generated samples
    generated_patterns = tf.random_uniform([num_samples, generator.num_units_per_layer[0]], 
                                 0.0, 1.0, "float32", seed=current_seed)

    generator.iteration(generated_patterns)
    discriminator.iteration(generator.units[-1])
    generated_outs = discriminator.units[-1]
    generated_probabilities = tf.tanh(generated_outs)

    # data samples
    data = tf.random_shuffle(data, current_seed)
    data_patterns = data[:num_samples,:]
    discriminator.iteration(data_patterns)
    data_outs = discriminator.units[-1]
    data_probabilities = tf.tanh(data_outs)     
 
    # Simulate
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        outs, douts, dprobs = session.run([discriminator.units[0], data_outs, data_probabilities])
        
        plt.imshow(outs[0].reshape(img_side, img_side))
        plt.show()
    print "Done"