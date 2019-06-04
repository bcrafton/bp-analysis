
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from conv import conv

#####

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
img = x_train[12] / 255.
img[0:14, 0:14] = 0.

img = np.reshape(img, (28, 28))
plt.imshow(img, cmap='gray')
plt.show()

#####
