"""
A classifier, called classify.py. Its first argument is the neural network to load up.
All remaining arguments are image files to classify. It should print out one line for
each file, with its name and whether it contains a cat or a dog. It may not make any
assumptions about file names.
"""

__author__ = "Emilee Oquist"
__license__ = "MIT"
__date__ = "April 2023"

import sys
import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

if __name__ == "__main__":
    if(len(sys.argv) < 3):
        print('Usage: python classify.py <neural network> <image file to classify>\n', file=sys.stderr)
        # python classify.py continue_training.h5 /Users/ecmo/cs431-cat-dog-nn/cats-and-dogs/c1.jpg
        exit(1)

    # Load the trained model
    model = load_model(sys.argv[1])
    print("model loaded") # !!!

    for iterator, filename in enumerate(sys.argv):
        if iterator >= 2:
            image = cv2.imread(filename)

            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
                image = cv2.resize(image, (100, 100))
                image = image.astype('uint8') / 255.0
                image = np.expand_dims(image, axis=0)  # add batch dimension
                if len(image.shape) == 2: # convert greyscale images to color
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Classify the image
            prediction = model.predict(image)
            class_idx = np.argmax(prediction)
            print('Predicted:', class_idx)
