# [HWR-01] 必要なモジュールをインポートします。
# In [1]:

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.examples.tutorials.mnist import input_data

train_x = np.abs(np.random.normal(0.0, 1.0, 5 * 360 * 363)).reshape(5, 360 * 363)
train_t = np.zeros(5 * 100).reshape(5, 100)


for i in range(5):
    while True:
        a = np.abs(np.random.normal(0.0, 1.0))
        if a < 1: 
            break
    train_t[i][int(a * 100)] = 1

print (train_x, train_t)

with open('train_x.data', mode='wb') as f:
    pickle.dump(train_x, f)

with open('train_t.data', mode='wb') as f:
    pickle.dump(train_t, f)
