import batch
import numpy as np

vec = np.array(range(10))

bv = batch.Batch(vec)

for i in range(10):
    print (bv.next_batch(3))
