import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import math

def TrainGraph(session_in):
    for i in range(1001):
        batch = mnist.train.next_batch(batchSize)
        session_in.run(
        train_step, feed_dict={x: batch[0],
        true_y: batch[1],
        keep_prob: 0.5})

        if i % 100 == 0 and i != 0:
        trainAccuracy = sess.run(accuracy, feed_dict={x:batch[0],true_y:batch[1], keep_prob:1.0})
        print("step %d, training accuracy %g"%(i, trainAccuracy))

def TestGraph(session_in, data):
    testAccuracy = session_in.run(accuracy,
            feed_dict={x: mnist.test.images,
            true_y: mnist.test.labels,
            keep_prob: 1.0})
    print("test accuracy %g" % (testAccuracy))


def Fetch(mode):
    print("fetch!!!!")
    if mode in ("-f", "--fetch"):
        global mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    else
        assert False, "Something went wrong during data fetch..."
