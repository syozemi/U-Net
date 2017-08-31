import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import shutil
from libs import (get_variable, get_conv, get_bias, get_pool, conv_and_pool)



with open('', 'rb') as f:
    image = pickle.load(f)

with open('data/ncratio10', 'rb') as f:
    ncratio10 = pickle.load(f)



class CNN:
    def __init__(self):
        with tf.Graph().as_default():
            self.prepare_model()
            self.prepare_session()

    def prepare_model(self):
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 572 * 572])
            h_pool = tf.reshape(x, [-1,572,572,1])

        with tf.name_scope('contracting'):
            layers = 4
            b = 20
            h_array = OrderedDict()

            x = tf.placeholder(tf.float32, [None, 572 * 572])
            h_pool = tf.reshape(x, [-1, 572, 572, 1])

            for i in range(layers):
                if i == 0:
                    filter1 = get_variable([3, 3, 1, b])
                else:
                    filter1 = get_variable([3, 3, b // 2, b])
                h1 = get_conv(h_pool, filter1, 1, 'VALID')

                filter2 = get_variable([3, 3, b, b])
                h2 = get_conv(h1, filter2, 1, 'VALID')

                h_array[i] = h2
                h_pool = get_pool(h2, 2)
                b = b * 2

        with tf.name_scope('floor'):
            filter5_1 = get_variable([3, 3, b // 2, b])
            h5_1 = get_conv(h_pool, filter5_1, 1, 'VALID')

            filter5_2 = get_variable([3, 3, b, b])
            h_pool = get_conv(h5_1, filter5_2, 1, 'VALID')


        with tf.name_scope('expanding')
            for i in range(layers):
                filter5 = get_variable([2, 2, b // 2, b])
                h3 = get_deconv2(h_pool, filter5)

                hcat = get_concat(h_array[3 - i], h3)
                filter1 = get_variable([3, 3, b, b // 2])
                h1 = get_conv(hcat, filter1, 1, 'VALID')

                filter2 = get_variable([3, 3, b // 2, b // 2])
                h_pool = get_conv(h1, filter2, 1, 'VALID')

                b = b // 2

            filter1_3 = get_variable([1, 1, b, 2])
            h_pool = get_conv(h_pool, filter1_3, 1, 'VALID') 

        # with tf.name_scope('fully_connected'):
        #     h_pool_flat = tf.reshape(h_pool, [-1, 2 * 388 * 388])
        #
        #     num_units1 = 2 * 388 * 388
        #     num_units2 = 388 * 388
        #
        #     w2 = get_variable([num_units1, num_units2])
        #     b2 = get_bias([num_units2])
        #     hidden2 = tf.nn.relu(tf.matmul(h_pool_flat, w2) + b2)
        #
        # with tf.name_scope('dropout'):
        #     keep_prob = tf.placeholder(tf.float32)
        #     hidden2_drop = tf.nn.dropout(hidden2, keep_prob)
        #
        # with tf.name_scope('softmax'):
        #     num_class = 10
        #
        #     w0 = get_variable([num_units2, num_class])
        #     b0 = get_bias([num_class])
        #     p = tf.nn.softmax(tf.matmul(hidden2_drop, w0) + b0)
        #
        with tf.name_scope('optimizer'):
            t = tf.placeholder(tf.float32, [None, num_class])
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t,logits=p))
            train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
            correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.name_scope('evaluator'):
            correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.histogram("convolution_filters1", h_conv1)
        tf.summary.histogram("convolution_filters2", h_conv2)
        
        self.x, self.t, self.p, self.keep_prob = x, t, p, keep_prob
        self.train_step = train_step
        self.loss = loss
        self.accuracy = accuracy

    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        summary = tf.summary.merge_all()

        saver = tf.train.Saver()
        if os.path.isdir('/tmp/logs'):
            shutil.rmtree('/tmp/logs')
        writer = tf.summary.FileWriter("/tmp/logs", sess.graph)
        
        self.sess = sess
        self.summary = summary
        self.writer = writer
        self.saver = saver

cnn = CNN()

i = 0
for _ in range(100):
    i += 1
    cnn.sess.run(cnn.train_step,
             feed_dict={cnn.x:image, cnn.t:ncratio10, cnn.keep_prob:0.1})
    if i % 1 == 0:
        summary, loss_val, acc_val = cnn.sess.run([cnn.summary, cnn.loss, cnn.accuracy],
                feed_dict={cnn.x:image,
                           cnn.t:ncratio10,
                           cnn.keep_prob:1.0})
        print ('Step: %d, Loss: %f, Accuracy: %f'
               % (i, loss_val, acc_val))
        # cnn.saver.save(cnn.sess, os.path.join(os.getcwd(), 'cnn_session'), global_step=i)
        cnn.writer.add_summary(summary, i)
