#!/usr/bin/env python
import math
import numpy as np
import matplotlib.pyplot as plt


def getActivations(layer, image, stimuli, sess):
    units = sess.run(
        layer,
        feed_dict={image:np.reshape(stimuli, [1, 784], order='F')})
    plotNNFilter(units)


def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20, 20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i + 1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0, :, :, i], interpolation="nearest", cmap="gray")
        plt.show()
