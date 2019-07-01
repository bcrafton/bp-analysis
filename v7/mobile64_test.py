
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

filters_name = 'bp'

if filters_name == 'lel':
    jpg_name = 'lel.jpg'
    npy_name = 'mobile64_lel.npy'
    
elif filters_name == 'bp':
    jpg_name = 'bp.jpg'
    npy_name = 'mobile64_bp.npy'

else:
    assert(False)


weights = np.load(npy_name).item()

conv1_filters = weights['block1_conv']

conv2_dw_filters = weights['block2_conv_block_dw_conv_dw']
conv2_pw_filters = weights['block2_conv_block_pw_conv']

conv3_dw_filters = weights['block3_conv_block_dw_conv_dw']
conv3_pw_filters = weights['block3_conv_block_pw_conv']

conv4_dw_filters = weights['block4_conv_block_dw_conv_dw']
conv4_pw_filters = weights['block4_conv_block_pw_conv']

conv5_dw_filters = weights['block5_conv_block_dw_conv_dw']
conv5_pw_filters = weights['block5_conv_block_pw_conv']

conv6_dw_filters = weights['block6_conv_block_dw_conv_dw']
conv6_pw_filters = weights['block6_conv_block_pw_conv']

conv7_dw_filters = weights['block7_conv_block_dw_conv_dw']
conv7_pw_filters = weights['block7_conv_block_pw_conv']

conv8_dw_filters = weights['block8_conv_block_dw_conv_dw']
conv8_pw_filters = weights['block8_conv_block_pw_conv']

conv9_dw_filters = weights['block9_conv_block_dw_conv_dw']
conv9_pw_filters = weights['block9_conv_block_pw_conv']

conv10_dw_filters = weights['block10_conv_block_dw_conv_dw']
conv10_pw_filters = weights['block10_conv_block_pw_conv']

conv1_filters = np.absolute(conv1_filters)
conv2_dw_filters = np.absolute(conv2_dw_filters)

#####

X = tf.placeholder(tf.float32, [1, 64, 64, 3])

conv1 = tf.nn.conv2d(X, conv1_filters, [1,1,1,1], 'SAME')

conv2_dw = tf.nn.depthwise_conv2d(conv1,    conv2_dw_filters, [1,1,1,1], 'SAME')
'''
conv2_pw = tf.nn.conv2d(          conv2_dw, conv2_pw_filters, [1,1,1,1], 'SAME')
conv3_dw = tf.nn.depthwise_conv2d(conv2_pw, conv3_dw_filters, [1,1,1,1], 'SAME')
conv3_pw = tf.nn.conv2d(          conv3_dw, conv3_pw_filters, [1,1,1,1], 'SAME')

conv4_dw = tf.nn.depthwise_conv2d(conv3_pw, conv4_dw_filters, [1,2,2,1], 'SAME')
conv4_pw = tf.nn.conv2d(          conv4_dw, conv4_pw_filters, [1,1,1,1], 'SAME')
conv5_dw = tf.nn.depthwise_conv2d(conv4_pw, conv5_dw_filters, [1,1,1,1], 'SAME')
conv5_pw = tf.nn.conv2d(          conv5_dw, conv5_pw_filters, [1,1,1,1], 'SAME')

conv6_dw = tf.nn.depthwise_conv2d(conv5_pw, conv6_dw_filters, [1,2,2,1], 'SAME')
conv6_pw = tf.nn.conv2d(          conv6_dw, conv6_pw_filters, [1,1,1,1], 'SAME')
conv7_dw = tf.nn.depthwise_conv2d(conv6_pw, conv7_dw_filters, [1,1,1,1], 'SAME')
conv7_pw = tf.nn.conv2d(          conv7_dw, conv7_pw_filters, [1,1,1,1], 'SAME')

conv8_dw = tf.nn.depthwise_conv2d(conv7_pw, conv8_dw_filters, [1,1,1,1], 'SAME')
conv8_pw = tf.nn.conv2d(          conv8_dw, conv8_pw_filters, [1,1,1,1], 'SAME')
conv9_dw = tf.nn.depthwise_conv2d(conv8_pw, conv9_dw_filters, [1,1,1,1], 'SAME')
conv9_pw = tf.nn.conv2d(          conv9_dw, conv9_pw_filters, [1,1,1,1], 'SAME')

conv10_dw = tf.nn.depthwise_conv2d(conv9_pw,  conv10_dw_filters, [1,2,2,1], 'SAME')
conv10_pw = tf.nn.conv2d(          conv10_dw, conv10_pw_filters, [1,1,1,1], 'SAME')
# conv11_dw = tf.nn.depthwise_conv2d(conv10_pw, conv11_dw_filters, [1,1,1,1], 'SAME')
# conv11_pw = tf.nn.conv2d(          conv11_dw, conv11_pw_filters, [1,1,1,1], 'SAME')
'''

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

#####

x = np.load('imagenet64_example.npy')
x = np.reshape(x[0], (1, 64, 64, 3))
x = x / 255.
print (x[0, 0:8, 0:8, 0])

#####

[conv1, conv2_dw] = sess.run([conv1, conv2_dw], feed_dict={X: x})

assert(np.shape(conv1) == (1, 64, 64, 32))
print (conv1[0, 0:8, 0:8, 0])

assert(np.shape(conv2_dw) == (1, 64, 64, 32))
print (conv2_dw[0, 0:8, 0:8, 0])

print (conv1[0, 0:2, 0:2, 0])
print (conv2_dw_filters[1:3, 1:3, 0, 0])
print (np.sum(conv2_dw_filters[1:3, 1:3, 0, 0] * conv1[0, 0:2, 0:2, 0]))

#####











