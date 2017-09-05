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
f = open("../U-Net_Gsan/settings.yml", encoding='UTF-8')
settings = yaml.load(f)

with open('data/image572', 'rb') as f:
    image_x = pickle.load(f)

with open('data/nucleus_label', 'rb') as f:
    image_t = pickle.load(f)

image_x, image_t = batch.shuffle_image(image_x, image_t)

#後で消す
print(len(image_x))
num_data = settings["num_data"] ##訓練用データ数
num_test = settings["num_test"]
train_x = image_x[:num_data]
test_x = image_x[num_data:num_data + num_test]
train_t = image_t[:num_data]
test_t = image_t[num_data:num_data + num_test]

#UNETを初期化しています。
unet = unet.UNET(settings["input_sizex"], settings["input_sizey"], settings["num_class"], depth = settings["depth"], layers_default = settings["layers_default"])

Batch_x = batch.Batch(train_x)
Batch_t = batch.Batch(train_t)
Batch_num = settings["Batch_num"] ##バッチ数


i = 0
for _ in range(settings["learnig_times"]): ##学習回数
    i += 1
    batch_x = Batch_x.next_batch(Batch_num)
    batch_t = Batch_t.next_batch(Batch_num)
    unet.sess.run(unet.train_step,
            feed_dict={unet.x:batch_x, unet.t:batch_t, unet.keep_prob:settings["keep_prob"]})
    if i % 10 == 0:
        summary, loss_val, acc_val = unet.sess.run([unet.summary, unet.loss, unet.accuracy],
                feed_dict={unet.x:test_x,
                           unet.t:test_t,
                           unet.keep_prob:1.0})
        print ('Step: %d, Loss: %.12f, Accuracy: %.12f'
               % (i, loss_val, acc_val))

        if os.path.isdir('saver') == False:
            os.mkdir('saver')
        unet.saver.save(unet.sess, os.path.join(os.getcwd(), 'saver/tmp/unet_session'), global_step=i)
        unet.writer.add_summary(summary, i)


num_image = settings["num_image"]
output_sizex = unet.output_sizex
output_sizey = unet.output_sizey
num_class = unet.num_class

timage = np.array(unet.sess.run([unet.tout], feed_dict = {unet.x:test_x, unet.t:test_t, unet.keep_prob:1.0}))
timage = timage.reshape(-1, output_sizex, output_sizey, num_class)
result = np.array(unet.sess.run([unet.result], feed_dict = {unet.x:test_x, unet.t:test_t, unet.keep_prob:1.0}))
result = result.reshape(-1, output_sizex, output_sizey, num_class)
result_image = np.zeros(result.size).reshape(-1, output_sizex, output_sizey, num_class)

for i in range(len(result)):
    for j in range(output_sizex):
        for k in range(output_sizey):
            at = np.argmax(result[i][j][k])
            result_image[i][j][k][at] = 1

fig = plt.figure(figsize = (12, 60))
for i in range(num_image):
    subplot = fig.add_subplot(num_image, 3, i * 3 + 1)
    subplot.imshow(timage[i,...,0], cmap = 'gray')
    subplot = fig.add_subplot(num_image, 3, i * 3 + 2)
    subplot.imshow(result_image[i,...,0], cmap = 'gray')
    subplot = fig.add_subplot(num_image, 3, i * 3 + 3)
    subplot.imshow(test_x[i,:,:], cmap = 'gray')

plt.show()
