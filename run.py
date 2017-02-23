
#!/usr/bin/env python
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from time import sleep
import numpy as np
import model as mod
import plotter as myplot
from random import randint

def TestVisualization(test_num, mnist, sess, model):
    for j in range(test_num):
        randidx = randint(1,10000)
        print("Test case: %d, showing test image #%d"%(j+1, randidx))

        imageToUse = mnist.test.images[randidx] ##mnist max train size: 10K
        plt.imshow(np.reshape(imageToUse,[28,28]), interpolation="nearest", cmap="gray")
        plt.show(block=False)
        sleep(5)
        plt.close()

        print("showing #1 hidden layer activation, image #%d"%(randidx))
        myplot.getActivations(model.hidden_1, model.image, imageToUse, sess)
        print("showing #2 hidden layer activation, image #%d"%(randidx))
        myplot.getActivations(model.hidden_2, model.image, imageToUse, sess)
        print("showing #3 hidden layer activation, image #%d"%(randidx))
        myplot.getActivations(model.hidden_3, model.image, imageToUse, sess)

def main():
    mnist = input_data.read_data_sets('./mnist/', one_hot=True)
    image = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])
    model = mod.Model(image, label)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    batch_size = 50
    test_num = 5

    for i in range(1001): #mnist max train size: 50K
            images, labels = mnist.train.next_batch(batch_size)
            sess.run(model.optimize, {image: images, label: labels})
            if i % 100 == 0 and i != 0:
                error = sess.run(model.error, {image: images, label: labels})
                print("step %d, training accuracy %g"%(i, error))

    TestVisualization(test_num, mnist, sess, model)
if __name__ == '__main__':
  main()
