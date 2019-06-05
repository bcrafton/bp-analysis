
import numpy as np
np.set_printoptions(precision=2)

import tensorflow as tf
import matplotlib.pyplot as plt
from conv import conv
from combine_filter import combine_filter

#####

(x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
img = x_train[12] / 255.
img = np.reshape(img, (1, 32, 32, 3))

#####

f1 = np.random.uniform(low=-1., high=1., size=(3, 3, 3, 32))
f2 = np.random.uniform(low=-1., high=1., size=(3, 3, 32, 64))

#####

out1 = conv(img,  f1, [1,1], 'same')
out1 = conv(out1, f2, [1,1], 'same')

out1 = np.reshape(out1, (1, 32, 32, 64))

#####

f = combine_filter(f1, f2, stride=1)
assert(np.shape(f) == (5, 5, 3, 64))

out2 = conv(img,  f, [1,1], 'same')
out2 = np.reshape(out2, (1, 32, 32, 64))

#####

print (np.all(out1 - out2 < 1e-4))
print (np.max(out1 - out2))
print (np.max(out1), np.max(out2))

#####

mask = (out1[0, :, :, 0] - out2[0, :, :, 0]) > 1e-3
# print (mask)
# print (mask * (out1[0, :, :, 0] - out2[0, :, :, 0]))

