import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

import batch
import unet #UNETをimportします

# np.random.seed(1919114)
# tf.set_random_seed(1919114)

with open('data/image572', 'rb') as f:
    image_x = pickle.load(f)

with open('data/nucleus_label', 'rb') as f:
    image_t = pickle.load(f)

image_x, image_t = batch.shuffle_image(image_x, image_t)

num_data = int(len(image_x)*0.8) ##訓練用データ数
train_x = image_x[:num_data]
test_x = image_x[num_data:]
train_t = image_t[:num_data]
test_t = image_t[num_data:]

#UNETを初期化しています。
unet = unet.UNET(572, 572, 2, depth = 4, layers_default = 8)

Batch_x = batch.Batch(train_x)
Batch_t = batch.Batch(train_t)
Batch_num = 5 ##バッチ数


i = 0
for _ in range(100): ##学習回数
    i += 1
    batch_x = Batch_x.next_batch(Batch_num)
    batch_t = Batch_t.next_batch(Batch_num)
    unet.sess.run(unet.train_step,
            feed_dict={unet.x:batch_x, unet.t:batch_t, unet.keep_prob:0.1})
    if i % 10 == 0:
        summary, loss_val, acc_val = unet.sess.run([unet.summary, unet.loss, unet.accuracy],
                feed_dict={unet.x:test_x,
                           unet.t:test_t,
                           unet.keep_prob:1.0})
        print ('Step: %d, Loss: %f, Accuracy: %f'
               % (i, loss_val, acc_val))

        if os.path.isdir('saver') == False:
            os.mkdir('saver')
        unet.saver.save(unet.sess, os.path.join(os.getcwd(), 'saver/tmp/unet_session'), global_step=i)
        unet.writer.add_summary(summary, i)


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
for i in range(len(result)):
    subplot = fig.add_subplot(len(result), 3, i * 3 + 1)
    subplot.imshow(timage[i,...,0], cmap = 'gray')
    subplot = fig.add_subplot(len(result), 3, i * 3 + 2)
    subplot.imshow(result_image[i,...,0], cmap = 'gray')
    subplot = fig.add_subplot(len(result), 3, i * 3 + 3)
    subplot.imshow(test_x[i,:,:], cmap = 'gray')

plt.show()
