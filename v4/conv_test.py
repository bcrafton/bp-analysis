
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from conv import conv

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

#####

out3 = np.reshape(out3, (1, 4, 4, 256))
out3 = out3 / np.max(out3)

plt.imshow(out3[0, :, :, 0:3], cmap='gray')
plt.show()

#####
