
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

f1 = np.load('cifar10_weights.npy').item()['conv1']
f2 = np.load('cifar10_weights.npy').item()['conv2']
f3 = np.load('cifar10_weights.npy').item()['conv3']

#####

out1 = conv(img,  f1, [2,2], 'same')
out2 = conv(out1, f2, [2,2], 'same')
out3 = conv(out2, f2, [2,2], 'same')

#####
'''
f = combine_filter(f1, f2, stride=2)
assert(np.shape(f) == (7, 7, 3, 64))

out2 = conv(img,  f, [2,2], 'valid')
out2 = np.reshape(out2, (1, 13, 13, 64))

#####

print (np.all(out1 - out2 < 1e-4))
print (np.max(out1 - out2))
print (np.max(out1), np.max(out2))
'''
