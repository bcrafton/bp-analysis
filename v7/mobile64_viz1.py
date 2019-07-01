
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

accum = conv1_filters
conv1 = accum

accum = combine_filter_dw(accum, conv2_dw_filters, stride=1); print (np.shape(accum))
conv2_dw = accum

'''
accum = combine_filter   (accum, conv2_pw_filters, stride=1); print (np.shape(accum))
accum = combine_filter_dw(accum, conv3_dw_filters, stride=1); print (np.shape(accum))
accum = combine_filter   (accum, conv3_pw_filters, stride=1); print (np.shape(accum))

accum = combine_filter_dw(accum, conv4_dw_filters, stride=4); print (np.shape(accum))
accum = combine_filter   (accum, conv4_pw_filters, stride=1); print (np.shape(accum))
accum = combine_filter_dw(accum, conv5_dw_filters, stride=1); print (np.shape(accum))
accum = combine_filter   (accum, conv5_pw_filters, stride=1); print (np.shape(accum))

accum = combine_filter_dw(accum, conv6_dw_filters, stride=8); print (np.shape(accum))
accum = combine_filter   (accum, conv6_pw_filters, stride=1); print (np.shape(accum))
accum = combine_filter_dw(accum, conv7_dw_filters, stride=1); print (np.shape(accum))
accum = combine_filter   (accum, conv7_pw_filters, stride=1); print (np.shape(accum))

accum = combine_filter_dw(accum, conv8_dw_filters, stride=8); print (np.shape(accum))
accum = combine_filter   (accum, conv8_pw_filters, stride=1); print (np.shape(accum))
accum = combine_filter_dw(accum, conv9_dw_filters, stride=1); print (np.shape(accum))
accum = combine_filter   (accum, conv9_pw_filters, stride=1); print (np.shape(accum))

accum = combine_filter_dw(accum, conv10_dw_filters, stride=16); print (np.shape(accum))
accum = combine_filter   (accum, conv10_pw_filters, stride=1); print (np.shape(accum))
'''

#####

x = np.load('imagenet64_example.npy')
x = np.reshape(x[0], (1, 64, 64, 3))
x = x / 255.
print (x[0, 0:8, 0:8, 0])

#####

conv1 = conv(x=x, filters=conv1, strides=[1,1], padding='same')
assert(np.shape(conv1) == (1, 64, 64, 32))
print (conv1[0, 0:8, 0:8, 0])

conv2_dw = conv(x=x, filters=conv2_dw, strides=[1,1], padding='same')
assert(np.shape(conv2_dw) == (1, 64, 64, 32))
print (conv2_dw[0, 0:8, 0:8, 0])

print (conv1[0, 0:2, 0:2, 0])
print (conv2_dw_filters[1:3, 1:3, 0, 0])
print (np.sum(conv2_dw_filters[1:3, 1:3, 0, 0] * conv1[0, 0:2, 0:2, 0]))
print ('-----')
print (x[0, 0:3, 0:3, :])
print (accum[2:5, 2:5, :, 0])
print (np.sum(x[0, 0:3, 0:3, :] * accum[2:5, 2:5, :, 0]))
print ('-----')



#####



