#!/usr/bin/env python
#  @package plotter
#  @author Attila Borcs
#
# Helper functions for activation layer visualization
# First the activation of a corresponding layer has been
# pulled out, then plotted with matplotlib

import math
import numpy as np
import matplotlib.pyplot as plt
import params as prm
from time import sleep

def getActivations(layer, image, stimuli, sess):
    units = sess.run(
        layer,
        feed_dict={image: np.reshape(stimuli, [1, mnist_img_vec], order='F')})
    plotNNFilter(units)

def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20, 20))
    n_rows = math.ceil(filters / prm.n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, prm.n_columns, i + 1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0, :, :, i], interpolation="nearest", cmap="gray")
    plt.show(block=False)
    sleep(prm.vis_delay)
    plt.close()
