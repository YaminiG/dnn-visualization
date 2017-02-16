
    def TrainGraph(session_in):
        for i in range(1001):
            batch = mnist.train.next_batch(batchSize)
            session_in.run(
                train_step, feed_dict={x: batch[0],
                                       true_y: batch[1],
                                       keep_prob: 0.5})
            if i % 100 == 0 and i != 0:
                trainAccuracy = session_in.run(
                    accuracy, feed_dict={x: batch[0],
                                         true_y: batch[1],
                                         keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, trainAccuracy))


    def TestGraph(session_in):
        testAccuracy = session_in.run(
            accuracy,
            feed_dict={x: mnist.test.images,
                       true_y: mnist.test.labels,
                       keep_prob: 1.0})
        print("test accuracy %g" % (testAccuracy))


        def ResetGraph(mode):

        if(mode)
            mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
            
        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, [None, 784], name="x-in")
        true_y = tf.placeholder(tf.float32, [None, 10], name="y-in")
        keep_prob = tf.placeholder("float")

        x_image = tf.reshape(x, [-1, 28, 28, 1])
        hidden_1 = slim.conv2d(x_image, 5, [5, 5])
        pool_1 = slim.max_pool2d(hidden_1, [2, 2])
        hidden_2 = slim.conv2d(pool_1, 5, [5, 5])
        pool_2 = slim.max_pool2d(hidden_2, [2, 2])
        hidden_3 = slim.conv2d(pool_2, 20, [5, 5])
        hidden_3 = slim.dropout(hidden_3, keep_prob)
        out_y = slim.fully_connected(
            slim.flatten(hidden_3), 10, activation_fn=tf.nn.softmax)

        cross_entropy = -tf.reduce_sum(true_y * tf.log(out_y))
        correct_prediction = tf.equal(tf.argmax(out_y, 1), tf.argmax(true_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        batchSize = 50

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        TrainGraph(sess)
