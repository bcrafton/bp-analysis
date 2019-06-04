
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from conv import conv

#####

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
img = x_train[12] / 255.

img = np.reshape(img, (1, 28, 28, 1))

#####

f1 = np.array([[1.,0.,-1.],[2.,0.,-2.],[1.,0.,-1.]])
f1 = np.reshape(f1, (3, 3, 1, 1))

f2 = np.array([[0.,1.,2.],[0.,2.,4.],[0.,1.,2.]])
f2 = np.reshape(f1, (3, 3, 1, 1))

#####

img1 = conv(img,  f1, [1,1], 'same')
img2 = conv(img1, f2, [1,1], 'same')
img2 = np.reshape(img2, (28, 28))
plt.imshow(img2, cmap='gray')
plt.show()

#####
