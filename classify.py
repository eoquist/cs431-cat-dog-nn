"""
A classifier, called classify.py. Its first argument is the neural network to load up.
All remaining arguments are image files to classify. It should print out one line for
each file, with its name and whether it contains a cat or a dog. It may not make any
assumptions about file names.
"""

__author__ = "Emilee Oquist"
__license__ = "MIT"
__date__ = "April 2023"

import os
import sys

if __name__ == "__main__":
    if(len(sys.argv) < 3):
        print('Usage: python classify.py <neural network> <image file to classify>\n', file=sys.stderr)
        exit(1)
    for i, arg in enumerate(sys.argv):
        print(f"Argument {i:>6}: {arg}")