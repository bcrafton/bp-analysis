
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from conv import conv

#####

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
img = x_train[12] / 255.

img = np.reshape(img, (1, 28, 28, 1))

#####

filter = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

filter = np.reshape(filter, (3, 3, 1, 1))

#####

out = conv(img, filter, [1,1], 'same')
out = np.reshape(out, (28, 28))
print (np.shape(out))
plt.imshow(out, cmap='gray')
plt.show()

#####
