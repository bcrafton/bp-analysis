
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from conv import conv
from combine_filter import combine_filter

#####

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
img = x_train[12] / 255.

img = np.reshape(img, (1, 28, 28, 1))

#####

f1 = np.array([[1.,0.,-1.],[2.,0.,-2.],[1.,0.,-1.]])
f2 = np.array([[0.,1.,2.],[0.,2.,4.],[0.,1.,2.]])

#####

f = combine_filter(f1, f2)
f = np.reshape(f, (5, 5, 1, 1))

#####

img1 = conv(img,  f, [1,1], 'valid')
img1 = np.reshape(img1, (24, 24))
plt.imshow(img1, cmap='gray')
plt.show()

#####
