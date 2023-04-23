"""
Neural Network Generator called make_nn.py
This program takes two command-line arguments:
    a directory containing images
    the name of the neural network file to create and save to disk
It assumes all training image file names begin with c or d in the create_training function.

NOTES:
My Tensorflow was not compiled to use AVX2 FMA -- non-optimized
"""
__author__ = "Emilee Oquist"
__license__ = "MIT"
__date__ = "April 2023"
# Credits to: Patrick Loeber and his CNN tutorial video

import os
import sys
import numpy as np
import pandas as pd
# from PIL import Image
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
# from keras.layers import Dense

# globals
image_directory = None
nn_model_filename = 'emilee.dnn'

test_images = None
test_labels = None

def create_training():
    """ Handling the give directory containing training images. """
    file_names = os.listdir(image_directory)
    # train_images = np.zeros((len(file_names), 100, 100, 3), dtype=np.uint8)
    train_labels = []

    for filename in file_names:
        # Create a list of labels for each image
        if filename.startswith('c'):
            train_labels.append(1)
        else:
            train_labels.append(0)

    print("len train_images: " + str(len(train_images)))
    print("len train_labels: " + str(len(train_images)))
    
    train_datagen = ImageDataGenerator(rescale=1./255., validation_split=0.2, dtype='uint8')
    dataframe = pd.DataFrame({
        'filename': file_names,
        'label': train_labels
    })

    # Mostly helps to batch feed
    train_generator = train_datagen.flow_from_dataframe( # DirectoryIterator
        dataframe=dataframe,
        directory=image_directory,
        filename='filename',
        label='label',
        batch_size=32,
        class_mode='binary',
        subset='training'
    )


    
    return train_images, train_labels

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

    # training
    batch_size = 64
    epochs = 5

    model.fit(train_images, train_labels, epochs=epochs,
            batch_size=batch_size, verbose=2)

    # evaulate
    model.evaluate(test_images,  test_labels, batch_size=batch_size, verbose=2)


    # SavedModel or HDF5 --> don't save it as a dir or save optimizer info
    # model.save("nn.h5")  # .h5 = HDF5 !!! its being saved as savedmodel instead of as h5 ??? verify
    model.save(nn_model_filename, include_optimizer = False)
    # This means the model can't be trained anymore, but it will lead to a more manageable file size.
    return model



# Memory will be at a premium during this assignment. It may help you to monitor your 
# computer’s free memory while you are executing this code. 
# Converting a list to an array, may be impossible to do in a reasonable amount of time.


# Beware of local minima while performing the gradient descent! If after a couple epochs,
# the accuracy is not improving, stop your program and try again. 
# Some networks may get stuck classifying everything as [X] or [Y], thus always getting 50% accuracy

if __name__ == "__main__":
    """ Handles command line arguments: an image directory and string name of the neural network file to create."""
    if(len(sys.argv) > 3):
        print('Usage: python make_nn.py <image directory> <name of neural network file to create> \n', file=sys.stderr)
        exit(1)

    image_directory = sys.argv[1] # /Users/ecmo/cs431-cat-dog-nn/cats-and-dogs
    nn_model_filename = sys.argv[2]

    # train_images = None
    # test_images = None

    # !!! DATA SET BEING MADE AND SEPARATED AS NECESSARY HERE !!!
    # Grayscale images won't appear to have the same dimensionality as color images.
    # They will appear to be 100 ×100, not 100 ×100 ×1 or 100 ×100 ×3
    # Find a way to change them to be “color” images before training on them.
    train_images, train_labels = create_training()
    