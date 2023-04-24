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
import time

import os
import sys
import numpy as np
import cv2
# import pandas as pd
# from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
# from sklearn.model_selection import train_test_split

# from keras.layers import Dense

# globals
image_directory = None
nn_model_filename = 'emilee.dnn'

test_images = None
test_labels = None

def create_training():
    """ Handling the give directory containing training images. """
    file_names = os.listdir(image_directory)
    num_files = len(file_names)
    train_images = np.zeros((num_files, 100, 100, 3), dtype=np.uint8)
    train_labels = np.zeros(len(file_names))

    start = time.time() # !!!
    for iterator, filename in enumerate(file_names):
        image = cv2.imread(os.path.join(image_directory, filename))
        if image is not None:
            if len(image.shape) == 2: # convert greyscale images to color
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # img = img.astype('uint8') / 255.0
            train_images[iterator] = image
            train_labels[iterator] = 1 if filename.startswith('c') else 0
    end = time.time() # !!!
    print("create training set time")
    print(end - start) # !!!

    # test set
    permutation = np.random.permutation(num_files)
    shuffled_images = train_images[permutation]
    shuffled_labels = train_labels[permutation]
    # Split the data into training and testing sets
    train_images = shuffled_images[:800]
    train_labels = shuffled_labels[:800]
    test_images = shuffled_images[800:]
    test_labels = shuffled_labels[800:]

    return train_images, train_labels, test_images, test_labels


def build_model():
    """
    Simple model for univariate regression (only one output neuron), with the given input shape 
    and the given number of hidden layers and neurons, and it compiles it using an optimizer 
    configured with the specified learning rate.
    """
    # Load the training and testing set
    train_images, train_labels, test_images, test_labels = create_training()

    model = Sequential([
        # Convolutional Layers
        # 32 filters, 3x3 kernel
        layers.Conv2D(32, (3, 3), strides=(2,2), padding="same", activation='relu', input_shape=(100, 100, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.5),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        # Dense Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
        # “softmax” layer that is trained on [1,0] for cats and [0,1] for dogs (or vice-versa). 
        # dense sigmoid allegedly better for binary classification over the general softmax
        # layers.Softmax()
    ])
    print(model.summary())

    # loss and optimizer
    model.compile(optimizer='adam',
              loss='binary_crossentropy', # only cat vs. dog
              metrics=['accuracy'])

    # training
    batch_size = 64
    epochs = 5

    model.fit(train_images, train_labels, epochs=epochs,
            batch_size=batch_size, verbose=2)

    # evaulate
    model.evaluate(test_images,  test_labels, batch_size=batch_size, verbose=2)


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
    if(len(sys.argv) < 3):
        print('Usage: python make_nn.py <image directory> <name of neural network file to create> \n', file=sys.stderr)
        exit(1)
    
    # python make_nn.py /Users/ecmo/cs431-cat-dog-nn/cats-and-dogs eoquist
    image_directory = sys.argv[1] 
    nn_model_filename = sys.argv[2]
    nn_model_filename += ".h5"

    # train_images = None
    # test_images = None

    # Grayscale images won't appear to have the same dimensionality as color images.
    # They will appear to be 100 ×100, not 100 ×100 ×1 or 100 ×100 ×3
    # Find a way to change them to be “color” images before training on them.

    # train_images, train_labels = create_training()
    build_model()
    