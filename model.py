# Working example for my blog post at:
# https://danijar.github.io/structuring-your-tensorflow-models
#!/usr/bin/env python
import functools
import tensorflow as tf
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.

    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class Model:

    def __init__(self, image, label):
        self.image = image
        self.label = label
        self.prediction
        self.optimize
        self.error
        self.hidden_1
        self.hidden_2
        self.hidden_3

    @define_scope(initializer=slim.xavier_initializer())
    def prediction(self):
        x = self.image
        x_image = tf.reshape(x,[-1,28,28,1])
        self.hidden_1 = slim.conv2d(x_image,5,[5,5])
        pool_1 = slim.max_pool2d(self.hidden_1,[2,2])
        self.hidden_2 = slim.conv2d(pool_1,5,[5,5])
        pool_2 = slim.max_pool2d(self.hidden_2,[2,2])
        hidden_3 = slim.conv2d(pool_2,20,[5,5])
        self.hidden_3 = slim.dropout(hidden_3,1.0)
        x = slim.fully_connected(slim.flatten(self.hidden_3),10,
        activation_fn=tf.nn.softmax)
        return x

    @define_scope
    def optimize(self):
        logprob = tf.log(self.prediction + 1e-12)
        cross_entropy = -tf.reduce_sum(self.label * logprob)
        optimizer = tf.train.AdamOptimizer(1e-4)
        return optimizer.minimize(cross_entropy)

    @define_scope
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))
