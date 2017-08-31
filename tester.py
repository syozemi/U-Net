import numpy as np
import matplotlib.pyplot as plt
import os
import process_data as processer
import tensorflow as tf
from collections import OrderedDict
from libs import (get_variable, get_conv, get_bias, get_pool, get_deconv2,
        get_concat, conv_and_pool)


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

filter5_1 = get_variable([3, 3, b // 2, b])
h5_1 = get_conv(h_pool, filter5_1, 1, 'VALID')

filter5_2 = get_variable([3, 3, b, b])
h_pool = get_conv(h5_1, filter5_2, 1, 'VALID')


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

images = np.random.rand(10 * 572 * 572).reshape(10, 572 * 572)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
h_poolp = sess.run(h_pool,feed_dict={x:images})
print (h_poolp.shape)
