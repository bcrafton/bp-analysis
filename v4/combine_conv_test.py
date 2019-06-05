
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from conv import conv
from combine_filter import combine_filter

#####

(x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
img = x_train[12] / 255.
img = np.reshape(img, (1, 32, 32, 3))

#####

f1 = np.random.uniform(low=-1., high=1., size=(5, 5, 3, 32))
f2 = np.random.uniform(low=-1., high=1., size=(3, 3, 32, 64))

#####

out1 = conv(img,  f1, [2,2], 'same')
out1 = conv(out1, f2, [1,1], 'same')

out1 = np.reshape(out1, (1, 16, 16, 64))

#####

f = combine_filter(f1, f2, stride=2)
assert(np.shape(f) == (9, 9, 3, 64))

out2 = conv(img,  f, [2,2], 'same')
out2 = np.reshape(out2, (1, 16, 16, 64))

#####

print (np.all(out1 - out2 < 1e-4))
print (np.max(out1 - out2))
print (np.max(out1), np.max(out2))
