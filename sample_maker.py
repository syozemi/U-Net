# [HWR-01] 必要なモジュールをインポートします。
# In [1]:

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 100

train_x = np.random.rand(batch_size * 360 * 360).reshape(batch_size, 360 * 360)
train_t = np.zeros(batch_size * 10).reshape(batch_size, 10)


for i in range(batch_size):
    a = np.random.rand()
    train_t[i][int(a * 10)] = 1

print (train_x, train_t)

with open('train_x.data', mode='wb') as f:
    pickle.dump(train_x, f)

with open('train_t.data', mode='wb') as f:
    pickle.dump(train_t, f)
