# ensure python version is being selected via the commmand palette
"""
Neural Network Generator called make_nn.py
This program takes two command-line arguments:
    a directory containing images
    the name of the neural network file to create and save to disk
It may assume that all training image file names begin with c or d, for “cat” and “dog” respectively.


NOTES:
option to save as an h5 or hd5
dont save it as a folder/dir
=====
Training a model will take your computer hours to do. It’s a good idea to let your
computer do this overnight. Also, if you have a desktop computer at home, this may
be faster than your laptop.
"""
# import os
import sys
# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential, Model
# from keras.preprocessing.image import ImageDataGenerator
# from keras.layers import Dense

# !!! DATA SET BEING MADE AND SEPARATED AS NECESSARY HERE !!!
# Grayscale images won't appear to have the same dimensionality as color images.
# They will appear to be 100 ×100, not 100 ×100 ×1 or 100 ×100 ×3
# Find a way to change them to be “color” images before training on them.
my_dataset = keras.datasets.cifar10
(train_images, train_labels), (test_images,test_labels) = my_dataset.load_data()
print(train_images.shape())



if __name__ == "__main__":
    """ Handles command line arguments: an image directory and string name of the neural network file to create."""
    if(len(sys.argv)!=3):
        print('Usage: python make_nn.py <image directory> <name of neural network file to create> \n', file=sys.stderr)
        exit(1)
    for i, arg in enumerate(sys.argv):
        print(f"Argument {i:>6}: {arg}")

# Keras will let you know the current accuracy of your network on the training set as it trains.
# You will have to come up with your own architecture for your network, including:
# How many layers there are.
# The numbers and sizes of kernels.
# The dropout rate to use.
# The sizes of your fully connected (“dense”) layers.

# n_neurons should be the pixel size? maybe? verify ???
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    """
    Simple model for univariate regression (only one output neuron), with the given input shape 
    and the given number of hidden layers and neurons, and it compiles it using an optimizer 
    configured with the specified learning rate.
    """
    model = Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(layers.Dense(n_neurons,activation="relu"))
    model.add(layers.Dense(1))
    # However, your last layer will almost certainly be a “softmax” layer, that is trained on [1,0]
    # for cats and [0,1] for dogs (or vice-versa). 

    # When saving your neural network, don’t save any optimizer information. 
    # In Keras, this can be done with:
    # model.save(model file name, include optimizer = False)
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse",optimizer=optimizer)

    # This means that your saved model cannot be trained anymore, but it will also make your
    # file sizes much more manageable.
    return model


# Keras and TensorFlow assume that you will be working with NumPy arrays

# Memory will be at a premium during this assignment. It may help you to monitor your 
# computer’s free memory while you are executing this code. 
# Converting a list to an array, may be impossible to do in a reasonable amount of time.

# Use economical data type (“dtype”) for your NumPy arrays: float16 and uint8 
# No need for every pixel’s data to need 3 separate float64s (AKA Java doubles)!

# Beware of local minima while performing the gradient descent! If after a couple epochs,
# the accuracy is not improving, stop your program and try again. 
# Some networks may get stuck classifying everything as [X] or [Y], thus always getting 50% accuracy