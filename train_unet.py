import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

import batch
import unet #UNETをimportします

np.random.seed(1919114)
tf.set_random_seed(1919114)

with open('data/train_x', 'rb') as f:
    image_x = pickle.load(f)

with open('data/train_t', 'rb') as f:
    image_t = pickle.load(f)


#UNETを初期化しています。
unet = unet.UNET(572, 588, 2, depth = 4, layers_default = 8)

i = 0
Batch_x = batch.Batch(image_x)
Batch_t = batch.Batch(image_t)

for _ in range(1):
    i += 1
    batch_x = Batch_x.next_batch(10)
    batch_t = Batch_t.next_batch(10)
    unet.sess.run(unet.train_step,
            feed_dict={unet.x:image_x, unet.t:image_t, unet.keep_prob:0.1})
    if i % 1 == 0:
        summary, loss_val, acc_val = unet.sess.run([unet.summary, unet.loss, unet.accuracy],
                feed_dict={unet.x:image_x,
                           unet.t:image_t,
                           unet.keep_prob:1.0})
        print ('Step: %d, Loss: %f, Accuracy: %f'
               % (i, loss_val, acc_val))

        if os.path.isdir('saver') == False:
            os.mkdir('saver')
        unet.saver.save(unet.sess, os.path.join(os.getcwd(), 'saver/unet_session'), global_step=i)
        # unet.writer.add_summary(summary, i)


output_sizex = unet.output_sizex
output_sizey = unet.output_sizey
num_class = unet.num_class

timage = np.array(unet.sess.run([unet.tout], feed_dict = {unet.x:image_x, unet.t:image_t, unet.keep_prob:1.0}))
timage = timage.reshape(-1, output_sizex, output_sizey, num_class)
result = np.array(unet.sess.run([unet.result], feed_dict = {unet.x:image_x, unet.t:image_t, unet.keep_prob:1.0}))
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
    subplot.imshow(image_x[i,:,:], cmap = 'gray')

plt.show()
