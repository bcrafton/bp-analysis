
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from conv import conv

#####

(x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
img = x_train[12] / 255.
img = np.reshape(img, (1, 32, 32, 3))

#####

f1 = np.random.uniform(size=(3, 3, 3, 32))
f2 = np.random.uniform(size=(3, 3, 32, 64))

#####

out1 = conv(img,  f1, [1,1], 'valid')
out2 = conv(out1, f2, [1,1], 'valid')

out2 = np.reshape(out2, (1, 28, 28, 64))
out2 = out2 / np.max(out2)

plt.imshow(out2[0, :, :, 0:3], cmap='gray')
plt.show()

#####
