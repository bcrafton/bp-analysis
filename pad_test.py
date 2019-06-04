
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from conv import conv

#####

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
img = x_train[12] / 255.

img = np.reshape(img, (1, 28, 28, 1))

img = np.pad(img, [[0, 0], [5, 5], [5, 5], [0, 0]], mode='constant')

#####

img = np.reshape(img, (38, 38))
plt.imshow(img, cmap='gray')
plt.show()

#####
