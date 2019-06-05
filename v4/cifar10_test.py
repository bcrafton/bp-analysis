
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

f1 = np.load('cifar10_conv.npy').item()['conv1']
f2 = np.load('cifar10_conv.npy').item()['conv2']
f3 = np.load('cifar10_conv.npy').item()['conv3']

#####

out1 = conv(img,  f1, [2,2], 'same')
out2 = conv(out1, f2, [2,2], 'same')
out3 = conv(out2, f3, [2,2], 'same')

o1 = out3

#####

f4 = combine_filter(f1, f2, stride=2)
f5 = combine_filter(f4, f3, stride=2)
out1 = conv(img,  f5, [8,8], 'same')

o2 = out1

#####

print (np.all(o1 - o2 < 1e-4))
print (np.max(o1 - o2))
print (np.max(o1), np.max(o2))

