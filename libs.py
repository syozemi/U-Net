import tensorflow as tf
import numpy as np

np.random.seed(20160704)
tf.set_random_seed(20160704)

def get_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

def get_conv(images, _filter, shift, padding):
    return tf.nn.conv2d(images, _filter, strides = [1, shift, shift, 1], padding = padding)

def get_bias(shape):
    return tf.Variable(tf.constant(0.1, shape = shape))

def get_pool(images, n):
    return tf.nn.max_pool(images, ksize = [1, n, n, 1], strides = [1, n, n, 1], padding = 'VALID')

def get_deconv2(images, _filter):
    x_shape = tf.shape(images)
    output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] // 2])
    return tf.nn.conv2d_transpose(images, _filter, output_shape, strides = [1, 2, 2, 1], padding = 'VALID')

def get_crop(images, shape): #imagesをshapeのサイズに真ん中で切り抜く
    x_shape = tf.shape(images)
    size = [-1, shape[0], shape[1], -1]
    offsets = [0, (x_shape[1] - size[1]) // 2, (x_shape[2] - size[2]) // 2, 0]
    return tf.slice(images, offsets, size)


def get_concat(images1, images2): #x1 should be bigger
    x1_shape = tf.shape(images1)
    x2_shape = tf.shape(images2)
    shape = [x2_shape[1], x2_shape[2]]
    images1_crop = get_crop(images1, shape)
    return tf.concat([images1_crop, images2], 3)  
    

def conv_and_pool(images, now_filter, next_filter, pixel, shift): #return convoluted and pooled layers
    w_conv = get_variable([pixel, pixel, now_filter, next_filter])
    h_conv = get_conv(images, w_conv, shift, 'VALID')
    b_conv = get_bias([next_filter])
    h_conv_cutoff = tf.nn.relu(h_conv + b_conv)
    h_pool = get_pool(h_conv_cutoff, 2)
    return [h_conv, h_pool]
