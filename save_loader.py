import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import yaml

import batch
import unet #UNETをimportします

# np.random.seed(1919114)
# tf.set_random_seed(1919114)

# yaml形式の設定を読み込む
f = open("settings.yml", encoding='UTF-8')
settings = yaml.load(f)

with open('data/train_x', 'rb') as f:
    image_x = pickle.load(f)

with open('data/train_t', 'rb') as f:
    image_t = pickle.load(f)

image_x, image_t = batch.shuffle_image(image_x, image_t)

#後で消す
print(len(image_x))
num_test = settings["num_test"]
test_x = image_x[:num_test]
test_t = image_t[:num_test]

#UNETを初期化しています。
unet = unet.UNET(settings["input_sizex"], settings["input_sizey"], settings["num_class"], 
        depth = settings["depth"], layers_default = settings["layers_default"], saver_num = settings["saver_num"])


num_image = settings["num_image"]
output_sizex = unet.output_sizex
output_sizey = unet.output_sizey
num_class = unet.num_class

def image_drawer(arrays): #[-1,2]の配列を[-1,output_sizex, output_sizey, num_class]にしてからnum_classの次元を潰す
    images = arrays.reshape(-1, output_sizex, output_sizey, num_class)
    res = np.zeros(images.size).reshape(-1, output_sizex, output_sizey)
    for i in range(len(images)):
        for j in range(output_sizex):
            for k in range(output_sizey):
                at = np.argmax(images[i][j][k])
                res[i][j][k] = at / (num_class - 1.0)
    return res


t = np.array(unet.sess.run([unet.tout], feed_dict = {unet.x:test_x, unet.t:test_t, unet.keep_prob:1.0}))
t_image = image_drawer(t)
result = np.array(unet.sess.run([unet.result], feed_dict = {unet.x:test_x, unet.t:test_t, unet.keep_prob:1.0}))
result_image = image_drawer(result)

fig = plt.figure(figsize = (16, 25))
for i in range(num_image):
    subplot = fig.add_subplot(num_image, 3, i * 3 + 1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.imshow(t_image[i,...])
    subplot = fig.add_subplot(num_image, 3, i * 3 + 2)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.imshow(result_image[i,...])
    subplot = fig.add_subplot(num_image, 3, i * 3 + 3)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.imshow(test_x[i,:,:], cmap = 'gray')

plt.show()
