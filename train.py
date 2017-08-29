# [CNN-01] 必要なモジュールをインポートして、乱数のシードを設定します。
# In [1]:

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

np.random.seed(20160704)
tf.set_random_seed(20160704)
# [CNN-02] MNISTのデータセットを用意します。
# In [2]:

with open('train_x.data', mode='rb') as f:
    train_x = pickle.load(f)

with open('train_t.data', mode='rb') as f:
    train_t = pickle.load(f)

print (train_x.shape, train_t.shape)
image_sizex = 360
image_sizey = 363
num_level = 100


print (train_x.size, train_t.size)

num_filters1 = 32

x = tf.placeholder(tf.float32, [None, image_sizex * image_sizey])
x_image = tf.reshape(x, [-1,image_sizex,image_sizey,1])

W_conv1 = tf.Variable(tf.truncated_normal([120,120,1,num_filters1],
                                          stddev=0.1))
h_conv1 = tf.nn.conv2d(x_image, W_conv1,
                       strides=[1,1,1,1], padding='VALID')

b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_filters1]))
h_conv1_cutoff = tf.nn.relu(h_conv1 + b_conv1)

h_pool1 = tf.nn.max_pool(h_conv1_cutoff, ksize=[1,2,2,1],
                         strides=[1,2,2,1], padding='SAME')
# [CNN-04] 2段目の畳み込みフィルターとプーリング層を定義します。
# In [4]:

num_filters2 = 32

W_conv2 = tf.Variable(tf.truncated_normal([120,120,num_filters1,num_filters2],
                                stddev=0.1))
h_conv2 = tf.nn.conv2d(h_pool1, W_conv2,
                       strides=[1,1,1,1], padding='VALID')

b_conv2 = tf.Variable(tf.constant(0.1, shape=[num_filters2]))
h_conv2_cutoff = tf.nn.relu(h_conv2 + b_conv2)

h_pool2 = tf.nn.max_pool(h_conv2_cutoff, ksize=[1,2,2,1],
                         strides=[1,2,2,1], padding='SAME')
# [CNN-05] 全結合層、ドロップアウト層、ソフトマックス関数を定義します。
# In [5]:

h_pool2_flat = tf.reshape(h_pool2, [-1, 320])

num_units1 = 320
num_units2 = 128

w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
b2 = tf.Variable(tf.constant(0.1, shape=[num_units2]))
hidden2 = tf.nn.relu(tf.matmul(h_pool2_flat, w2) + b2)

keep_prob = tf.placeholder(tf.float32)
hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

w0 = tf.Variable(tf.zeros([num_units2, num_level]))
b0 = tf.Variable(tf.zeros([num_level]))
p = tf.nn.softmax(tf.matmul(hidden2_drop, w0) + b0)
# [CNN-06] 誤差関数 loss、トレーニングアルゴリズム train_step、正解率 accuracy を定義します。
# In [6]:

t = tf.placeholder(tf.float32, [None, num_level])
loss = -tf.reduce_sum(t * tf.log(p))
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# [CNN-07] セッションを用意して、Variable を初期化します。
# In [7]:

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
# [CNN-08] パラメーターの最適化を20000回繰り返します。
# 最終的に、テストセットに対して約99%の正解率が得られます。
# In [8]:

i = 0
for _ in range(40000):
    i += 1
    sess.run(train_step,
            feed_dict={x:train_x, t:train_t, keep_prob:0.5}) # keep_prob: dropout rate
    if i % 500 == 0:
        loss_val, acc_val = sess.run([loss, accuracy],
                feed_dict={x:train_x, t:train_t, keep_prob:1.0})
        print ('Step: %d, Loss: %f, Accuracy: %f'
               % (i, loss_val, acc_val))
        saver.save(sess, os.path.join(os.getcwd(), 'cnn_session'), global_step=i)
# Step: 500, Loss: 1539.889160, Accuracy: 0.955600
# Step: 1000, Loss: 972.987549, Accuracy: 0.971700
# Step: 1500, Loss: 789.961914, Accuracy: 0.974000
# Step: 2000, Loss: 643.896973, Accuracy: 0.978400
# Step: 2500, Loss: 602.963257, Accuracy: 0.980900
# Step: 3000, Loss: 555.896484, Accuracy: 0.981900
# Step: 3500, Loss: 457.530762, Accuracy: 0.985300
# Step: 4000, Loss: 430.855194, Accuracy: 0.987000
# Step: 4500, Loss: 404.523743, Accuracy: 0.986600
# Step: 5000, Loss: 407.742065, Accuracy: 0.987100
# Step: 5500, Loss: 374.555054, Accuracy: 0.988300
# Step: 6000, Loss: 382.756165, Accuracy: 0.986900
# Step: 6500, Loss: 355.421509, Accuracy: 0.988000
# Step: 7000, Loss: 355.007141, Accuracy: 0.988900
# Step: 7500, Loss: 327.024780, Accuracy: 0.989300
# Step: 8000, Loss: 340.774933, Accuracy: 0.988000
# Step: 8500, Loss: 347.032379, Accuracy: 0.988300
# Step: 9000, Loss: 311.977875, Accuracy: 0.990400
# Step: 9500, Loss: 337.671753, Accuracy: 0.988700
# Step: 10000, Loss: 319.527100, Accuracy: 0.989600
# Step: 10500, Loss: 293.324158, Accuracy: 0.990500
# Step: 11000, Loss: 288.691833, Accuracy: 0.990200
# Step: 11500, Loss: 294.355652, Accuracy: 0.990100
# Step: 12000, Loss: 308.601837, Accuracy: 0.990600
# Step: 12500, Loss: 300.200623, Accuracy: 0.989800
# Step: 13000, Loss: 294.467682, Accuracy: 0.991200
# Step: 13500, Loss: 273.863708, Accuracy: 0.991600
# Step: 14000, Loss: 282.099548, Accuracy: 0.990800
# Step: 14500, Loss: 274.422974, Accuracy: 0.991200
# Step: 15000, Loss: 269.755096, Accuracy: 0.991300
# Step: 15500, Loss: 273.898376, Accuracy: 0.991600
# Step: 16000, Loss: 253.827591, Accuracy: 0.991900
# Step: 16500, Loss: 273.175781, Accuracy: 0.991500
# Step: 17000, Loss: 278.549866, Accuracy: 0.990100
# Step: 17500, Loss: 278.320190, Accuracy: 0.991500
# Step: 18000, Loss: 258.416412, Accuracy: 0.991200
# Step: 18500, Loss: 285.394806, Accuracy: 0.990900
# Step: 19000, Loss: 290.716187, Accuracy: 0.991000
# Step: 19500, Loss: 272.024597, Accuracy: 0.991600
# Step: 20000, Loss: 269.107910, Accuracy: 0.991800
# [CNN-09] セッション情報を保存したファイルが生成されていることを確認します。
# In [9]:

# !ls cnn_session*
# cnn_session-18000	cnn_session-19000	cnn_session-20000
# cnn_session-18000.meta	cnn_session-19000.meta	cnn_session-20000.meta
# cnn_session-18500	cnn_session-19500
# cnn_session-18500.meta	cnn_session-19500.meta

