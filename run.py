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
import model as mod


def usage():
    print 'run.py -option'
    print '<option>: -f <Fetch Data from web...>'
    print '<option>: -w <Test on Webam...>'
    print '<option>: -m <Test on MNIST Data...>'

def main():
        try:
            opts, args = getopt.getopt(sys.argv[1:], "twh",
            ["train","webcam","help"])
        except getopt.GetoptError as err:

            print('<option is not recognized>')
            usage()
            sys.exit(2)

            for o, a in opts:
                if o in ("-t", "--train"):
                    print("train...%s"%(o))
                    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
                    image = tf.placeholder(tf.float32, [None, 784])
                    label = tf.placeholder(tf.float32, [None, 10])
                    model = mod.Model(image, label)

                    batchSize = 50
                    sess = tf.Session()
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    for i in range(1001):
                        batch = mnist.train.next_batch(batchSize)
                        sess.run(model.optimize, feed_dict={image:batch[0],label:batch[1], keep_prob:0.5})
                        if i % 100 == 0 and i != 0:
                            trainAccuracy = sess.run(model.error, feed_dict={image:batch[0],label:batch[1], keep_prob:1.0})
                            print("step %d, training accuracy %g"%(i, trainAccuracy))

                            #    for _ in range(10):
                            #            images, labels = mnist.test.images, mnist.test.labels
                            #                error = sess.run(model.error, {image: images, label: labels})
                            #                print('Test error {:6.2f}%'.format(100 * error))
                            #                    for _ in range(60):
                            #                            images, labels = mnist.train.next_batch(100)
                            #                        sess.run(model.optimize, {image: images, label: labels})

                        elif o in ("-w", "--webcam"):
                            print('Test on Webcam...')
                        elif o in ("-h", "--help"):
                            usage()
                            sys.exit()
                            #imageToUse = mnist.test.images[0]
                            #plt.imshow(
                            #        np.reshape(imageToUse, [28, 28]), interpolation="nearest", cmap="gray")
                            #        getActivations(hidden_1, imageToUse)
                            #            getActivations(hidden_2, imageToUse)
                            #    getActivations(hidden_3, imageToUse)

                        else:
                            assert False, "unhandled option"

if __name__ == "__main__":
   main()
