import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import shutil
from collections import OrderedDict
from libs import (get_variable, get_conv, get_bias, get_pool, get_crop, get_concat, get_deconv2)



with open('data/train_x', 'rb') as f:
    image_x = pickle.load(f)

with open('data/train_t', 'rb') as f:
    image_t = pickle.load(f)



class UNET: #画像はテストデータラベルともにフラットにして入力
    def __init__(self):
        with tf.Graph().as_default():
            self.prepare_model()
            self.prepare_session()

    def prepare_model(self):
        input_size = 572   #input_size % 16 == 12ののものなら何でもよい
        output_size = input_size - 184
        #テストデータ[-1, 572 * 572]
        #ラベル[-1, 572 * 572, 2]
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, input_size * input_size])
            h_pool = tf.reshape(x, [-1,input_size,input_size,1])

        with tf.name_scope('contracting'):
            layers = 4
            b = 2  #層を多くする。論文は64でやってる。
            num_class = 2
            h_array = OrderedDict()

            x = tf.placeholder(tf.float32, [None, input_size * input_size])
            h_pool = tf.reshape(x, [-1, input_size, input_size, 1])

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


        with tf.name_scope('expanding'):
            for i in range(layers):
                filter5 = get_variable([2, 2, b // 2, b])
                h3 = get_deconv2(h_pool, filter5)

                hcat = get_concat(h_array[3 - i], h3)
                filter1 = get_variable([3, 3, b, b // 2])
                h1 = get_conv(hcat, filter1, 1, 'VALID')

                filter2 = get_variable([3, 3, b // 2, b // 2])
                h_pool = get_conv(h1, filter2, 1, 'VALID')

                b = b // 2

            filter1_3 = get_variable([1, 1, b, num_class])
            h_pool = get_conv(h_pool, filter1_3, 1, 'VALID') 

        with tf.name_scope('softmax'):
            h_pool_flat = tf.reshape(h_pool, [-1, num_class])

            result = tf.nn.softmax(h_pool_flat)

        with tf.name_scope('optimizer'):
            t = tf.placeholder(tf.float32, [None, input_size * input_size, num_class])
            tr = tf.reshape(t, [-1, input_size, input_size, num_class])
            tcrop = get_crop(tr, [output_size, output_size])
            tout = tf.reshape(tcrop, [-1, num_class])
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tout,logits=result))
            train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

        with tf.name_scope('evaluator'):
            correct_prediction = tf.equal(tf.argmax(result, 1), tf.argmax(tout, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        
        self.x, self.t, self.result = x, t, result
        self.train_step = train_step
        self.loss = loss
        self.tout = tout
        self.result = result
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

unet = UNET()

i = 0
for _ in range(50):
    i += 1
    unet.sess.run(unet.train_step,
             feed_dict={unet.x:image_x, unet.t:image_t})
    if i % 10 == 0:
        summary, loss_val, acc_val = unet.sess.run([unet.summary, unet.loss, unet.accuracy],
                feed_dict={unet.x:image_x,
                           unet.t:image_t})
        print ('Step: %d, Loss: %f, Accuracy: %f'
               % (i, loss_val, acc_val))
        # unet.saver.save(unet.sess, os.path.join(os.getcwd(), 'unet_session'), global_step=i)
        unet.writer.add_summary(summary, i)

timage = np.array(unet.sess.run([unet.tout], feed_dict = {unet.x:image_x, unet.t:image_t}))
timage = timage.reshape(5, 388, 388, 2)
result = np.array(unet.sess.run([unet.result], feed_dict = {unet.x:image_x, unet.t:image_t}))
result = result.reshape(5, 388, 388, 2)
result_image = np.zeros(5 * 388 * 388 * 2).reshape(5, 388, 388, 2)
for i in range(5):
    for j in range(388):
        for k in range(388):
            if result[i][j][k][0] < result[i][j][k][1]:
                result_image[i][j][k][0] = 1
            else:
                result_image[i][j][k][1] = 1

fig = plt.figure(figsize = (1, 2))
subplot = fig.add_subplot(1, 2, 1)
subplot.imshow(timage[0,...,0])
subplot = fig.add_subplot(1, 2, 2)
subplot.imshow(result_image[0,...,0])

plt.show()
