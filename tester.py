import numpy as np
import matplotlib.pyplot as plt
import os
import process_data as processer

mat = np.zeros(11 * 10).reshape(11, 10)
mat_shape = mat.shape
for i in range(mat_shape[0]):
    for j in range(mat_shape[1]):
        mat[i][j] = i * mat_shape[1] + j
mat_crop = processer.crop(mat, [10, 10])
print (mat)
print (mat_crop)
