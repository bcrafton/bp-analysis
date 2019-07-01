
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--alg', type=str, default='bp')
args = parser.parse_args()

import numpy as np
np.set_printoptions(precision=2)
import tensorflow as tf
import matplotlib.pyplot as plt

from conv import conv

from combine_filter import combine_filter
from combine_filter import combine_filter_dw

from viz_filter import viz_filter
from viz_filter import viz_filter_3_channels
from viz_filter import viz_filter_1_channels

#####

if args.alg == 'lel0':
    jpg_name = 'lel0.jpg'
    npy_name = 'vgg64_lel0.npy'
    accum_name = 'vgg64_lel0_combined'
    
elif args.alg == 'lel1':
    jpg_name = 'lel1.jpg'
    npy_name = 'vgg64_lel1.npy'
    accum_name = 'vgg64_lel1_combined'

else:
    assert(False)

weights = np.load(npy_name).item()

conv1_filters = weights['block1_conv_block_conv']
conv2_filters = weights['block2_conv_block_conv']
conv3_filters = weights['block3_conv_block_conv']
conv4_filters = weights['block4_conv_block_conv']
conv5_filters = weights['block5_conv_block_conv']
conv6_filters = weights['block6_conv_block_conv']
conv7_filters = weights['block7_conv_block_conv']
conv8_filters = weights['block8_conv_block_conv']
conv9_filters = weights['block9_conv_block_conv']
conv10_filters = weights['block10_conv_block_conv']

#####

X = tf.placeholder(tf.float32, [1, 224, 224, 3])

conv1 = tf.nn.conv2d(X, conv1_filters, [1,1,1,1], 'VALID')
conv2 = tf.nn.conv2d(conv1, conv2_filters, [1,2,2,1], 'VALID')

conv3 = tf.nn.conv2d(conv2, conv3_filters, [1,1,1,1], 'VALID')
conv4 = tf.nn.conv2d(conv3, conv4_filters, [1,2,2,1], 'VALID')

conv5 = tf.nn.conv2d(conv4, conv5_filters, [1,1,1,1], 'VALID')
conv6 = tf.nn.conv2d(conv5, conv6_filters, [1,2,2,1], 'VALID')

conv7 = tf.nn.conv2d(conv6, conv7_filters, [1,1,1,1], 'VALID')
conv8 = tf.nn.conv2d(conv7, conv8_filters, [1,2,2,1], 'VALID')

conv9  = tf.nn.conv2d(conv8, conv9_filters, [1,1,1,1], 'VALID')
conv10 = tf.nn.conv2d(conv9, conv10_filters, [1,1,1,1], 'VALID')

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

#####

x = np.load('imagenet224_example.npy')
x = np.reshape(x[0], (1, 224, 224, 3))
x = x / 255.

#####

[out] = sess.run([conv10], feed_dict={X: x})
print (np.shape(out))
np.save('ref', out)

#####











