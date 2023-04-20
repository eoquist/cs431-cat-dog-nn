"""
Neural Network Generator called make_nn.py
This program takes two command-line arguments:
    a directory containing images
    the name of the neural network file to create and save to disk
It may assume that all training image file names begin with c or d, for “cat” and “dog” respectively.

NOTES:
Training a model will take your computer hours to do. It’s a good idea to let your
computer do this overnight. Also, if you have a desktop computer at home, this may
be faster than your laptop.
"""
__author__ = "Emilee Oquist"
__license__ = "MIT"
__date__ = "April 2023"
# Credits to: Patrick Loeber and his CNN tutorial video

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

image_directory = None
nn_model_filename = 'eoquist.dnn'
# model.save("nn.h5")  # .h5 = HDF5 !!! its being saved as savedmodel instead of as h5 ??? verify

# !!! DATA SET BEING MADE AND SEPARATED AS NECESSARY HERE !!!
# Grayscale images won't appear to have the same dimensionality as color images.
# They will appear to be 100 ×100, not 100 ×100 ×1 or 100 ×100 ×3
# Find a way to change them to be “color” images before training on them.
my_dataset = keras.datasets.cifar10
(train_images, train_labels), (test_images,test_labels) = my_dataset.load_data()
print(train_images.shape())
# normalize: 0,255 -> 0,1
train_images, test_images = train_images / 255.0, test_images / 255.0


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
    model.add(layers.Conv2D(32, (3,3), strides=(1,1), padding="valid", activation='relu', input_shape=(32,32,3)))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(32, 3, activation='relu'))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    # “softmax” layer that is trained on [1,0] for cats and [0,1] for dogs (or vice-versa). 
    model.add(layers.Softmax())
    print(model.summary())

    # loss and optimizer
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(lr=0.001)
    metrics = ["accuracy"]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # SavedModel or HDF5 --> don't save it as a dir or save optimizer info
    model.save(nn_model_filename, include_optimizer = False)
    # This means the model can't be trained anymore, but it will lead to a more manageable file size.
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

if __name__ == "__main__":
    """ Handles command line arguments: an image directory and string name of the neural network file to create."""
    if(len(sys.argv)!=3):
        print('Usage: python make_nn.py <image directory> <name of neural network file to create> \n', file=sys.stderr)
        exit(1)
    for i, arg in enumerate(sys.argv):
        print(f"Argument {i:>6}: {arg}")