import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import yaml

import batch
import unet #unetをimportします
import libs
import process_data as pro

# np.random.seed(1919114)
# tf.set_random_seed(1919114)

# yaml形式の設定を読み込む
f = open("settings.yml", encoding='UTF-8')
settings = yaml.load(f)

image_x, image_t = pro.load_data_unet()

image_x, image_t = libs.shuffle_data(image_x, image_t)

#後で消す
print(len(image_x))
num_data = settings["num_data"] ##訓練用データ数
num_test = settings["num_test"]
train_x = image_x[:num_data]
test_x = image_x[num_data:num_data + num_test]
train_t = image_t[:num_data]
test_t = image_t[num_data:num_data + num_test]

#unetを初期化しています。
unet1 = unet.UNET(settings["input_sizex"], settings["input_sizey"],
        settings["num_class"], 1, depth = settings["depth"], layers_default = settings["layers_default"])

Batch_x = batch.Batch(train_x)
Batch_t = batch.Batch(train_t)
Batch_num = settings["Batch_num"] ##バッチ数


i = 0
for _ in range(settings["learning_times"]): ##学習回数
    i += 1
    batch_x = Batch_x.next_batch(Batch_num)
    batch_t = Batch_t.next_batch(Batch_num)
    unet1.sess.run(unet1.train_step,
            feed_dict={unet1.x:batch_x, unet1.t:batch_t, unet1.keep_prob:settings["keep_prob"]})
    if i % 10 == 0:
        summary, loss_val, acc_val, loss1, loss2= unet1.sess.run([unet1.summary, unet1.loss, unet1.accuracy, unet1.loss1, unet1.loss2],
                feed_dict={unet1.x:test_x,
                           unet1.t:test_t,
                           unet1.keep_prob:1.0})
        print ('Step: %d, Loss: %.12f, Accuracy: %.12f, Loss1: %.12f, Loss2: %.12f'
               % (i, loss_val, acc_val, loss1, loss2))
        if os.path.isdir('saver') == False:
            os.mkdir('saver')
        unet1.saver.save(unet1.sess, os.path.join(os.getcwd(), 'saver/tmp/unet1_session'), global_step=i)
        unet1.writer.add_summary(summary, i)

result_t = unet1.sess.run(unet1.result_t, feed_dict={unet1.x:train_x, unet1.t:train_t, unet1.keep_prob:1.0})
train_x = np.append(train_x, result_t[...,1].reshape(result_t.shape[0], result_t.shape[1],result_t.shape[2], 1), axis = 3)
train_x = np.append(train_x, result_t[...,2].reshape(result_t.shape[0], result_t.shape[1],result_t.shape[2], 1), axis = 3)

result_t = unet1.sess.run(unet1.result_t, feed_dict={unet1.x:test_x, unet1.t:test_t, unet1.keep_prob:1.0})
test_x = np.append(test_x, result_t[...,1].reshape(result_t.shape[0], result_t.shape[1],result_t.shape[2], 1), axis = 3)
test_x = np.append(test_x, result_t[...,2].reshape(result_t.shape[0], result_t.shape[1],result_t.shape[2], 1), axis = 3)

unet2 = unet.UNET(settings["input_sizex"], settings["input_sizey"],
        settings["num_class"], 1 + 2, depth = settings["depth"], layers_default = settings["layers_default"])


print (train_x.shape, test_x.shape)
Batch_x = batch.Batch(train_x)
Batch_t = batch.Batch(train_t)
Batch_num = settings["Batch_num"] ##バッチ数

i = 0
for _ in range(settings["learning_times"]): ##学習回数
    i += 1
    batch_x = Batch_x.next_batch(Batch_num)
    batch_t = Batch_t.next_batch(Batch_num)
    unet2.sess.run(unet2.train_step,
            feed_dict={unet2.x:batch_x, unet2.t:batch_t, unet2.keep_prob:settings["keep_prob"]})
    if i % 10 == 0:
        summary, loss_val, acc_val, loss1, loss2 = unet2.sess.run([unet2.summary, unet2.loss, unet2.accuracy, unet2.loss1, unet2.loss2],
                feed_dict={unet2.x:test_x,
                           unet2.t:test_t,
                           unet2.keep_prob:1.0})
        print ('Step: %d, Loss: %.12f, Accuracy: %.12f, Loss1: %.12f, Loss2: %.12f'
               % (i, loss_val, acc_val, loss1, loss2))
        if os.path.isdir('saver') == False:
            os.mkdir('saver')
        unet2.saver.save(unet2.sess, os.path.join(os.getcwd(), 'saver/tmp/unet2_session'), global_step=i)
        unet2.writer.add_summary(summary, i)


num_image = settings["num_image"]
output_sizex = unet2.output_sizex
output_sizey = unet2.output_sizey
num_class = unet2.num_class

t = np.array(unet2.sess.run([unet2.tout], feed_dict = {unet2.x:test_x, unet2.t:test_t, unet2.keep_prob:1.0}))
t_image = libs.image_convert(t, output_sizex, output_sizey, num_class)
result = np.array(unet2.sess.run([unet2.result], feed_dict = {unet2.x:test_x, unet2.t:test_t, unet2.keep_prob:1.0}))
result_image = libs.image_convert(result, output_sizex, output_sizey, num_class)

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
    subplot.imshow(test_x[i,...,0], cmap = 'gray')

plt.savefig("saver/tmp/prediction.png")
plt.show()
