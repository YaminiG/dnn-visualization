#!/usr/bin/python
#  @package run
#  @author Attila Borcs
#
#  Main function that runs the entier algorithm
#  This section fetching the MNIST training data
#  followed by the instantiation and the training
#  of tensorflow graph. Finally, the visualization
#  of the neural network is executed.

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from time import sleep
import numpy as np
import model as mod
import plotter as myplot
import params as prm
from random import randint

def TestVisualization(test_num, mnist, sess, model):
    """
    DNN Visualization done by randomly selecting predefined number
    (I used 5) of test cases (images) from the MNIST test database.
    The TestVisualization function automatically runs through all
    of the hidden layers and plot its activation, and wait 5 sec before
    before showing the next layer.
    """
    for j in range(test_num):
        randidx = randint(prm.mnist_min_size, prm.mnist_max_size)
        print("Test case: %d, showing test image #%d" % (j + 1, randidx))

        imageToUse = mnist.test.images[randidx]
        plt.imshow(
            np.reshape(imageToUse, [prm.mnist_img_size, prm.mnist_img_size]),
            interpolation="nearest",
            cmap="gray")
        plt.show(block=False)
        sleep(prm.vis_delay)
        plt.close()

        print("showing #1 hidden layer activation, image #%d" % (randidx))
        myplot.getActivations(model.hidden_1, model.image, imageToUse, sess)
        print("showing #2 hidden layer activation, image #%d" % (randidx))
        myplot.getActivations(model.hidden_2, model.image, imageToUse, sess)
        print("showing #3 hidden layer activation, image #%d" % (randidx))
        myplot.getActivations(model.hidden_3, model.image, imageToUse, sess)

def main():
    mnist = input_data.read_data_sets('./mnist/', one_hot=True)
    image = tf.placeholder(tf.float32, [None, prm.mnist_img_vec])
    label = tf.placeholder(tf.float32, [None, prm.mnist_label_size])
    model = mod.Model(image, label)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(prm.iter_num):
        images, labels = mnist.train.next_batch(prm.batch_size)
        sess.run(model.optimize, {image: images, label: labels})
        if i % 100 == 0 and i != 0:
            error = sess.run(model.error, {image: images, label: labels})
            print("step %d, training accuracy %g" % (i, error))

    TestVisualization(prm.test_num, mnist, sess, model)

if __name__ == '__main__':
    main()
