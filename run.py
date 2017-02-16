#!/usr/bin/env python
## @package dnn-visualization
#  @author Attila Borcs
#
# Descripction: Visualization Tool for debugging layer activations
# from convulitional layers.
#
# Used packages: Tensorflow, Opencv
#
# The approach utilized in this code originally pushlished
# in the Deep Learning Workshop, 31 st International
# Conference on Machine Learning, Lille, France, 2015.
#
# "Understanding Neural Networks Through Deep Visualization" by
# Jason Yosinski, Jeff Clune, Anh Nguyen, Thomas Fuchs, and Hod Lipson
# Direct link: https://arxiv.org/pdf/1506.06579.pdf

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import math
import sys
import getopt

def usage():
     print 'run.py -option'
     print '<option>: -f <Fetch Data from web...>'
     print '<option>: -w <Test on Webam...>'
     print '<option>: -m <Test on MNIST Data...>'

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "fhw", ["fetch", "help", "webcam"])
    except getopt.GetoptError as err:

        print('<option is not recognized>')
        usage()
        sys.exit(2)

    parse_data = False
    for o, a in opts:
        if o in ("-f", "--fetch"):
            parse_data = True
            print('Parsing Data...')
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-w", "--webcam"):
            print('Test on Webam...')
        else:
            assert False, "unhandled option"
            usage()


if __name__ == "__main__":
    main()

#if __name__ == "__main__":

#TODO: plug webcam images here
#imageToUse = mnist.test.images[0]

#plt.imshow(
#    np.reshape(imageToUse, [28, 28]), interpolation="nearest", cmap="gray")

#getActivations(hidden_1, imageToUse)
#getActivations(hidden_2, imageToUse)
#getActivations(hidden_3, imageToUse)
