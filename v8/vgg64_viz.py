
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--alg', type=str, default='bp')
args = parser.parse_args()

import numpy as np
np.set_printoptions(precision=2)
import scipy
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

elif args.alg == 'bp':
    jpg_name = 'bp.jpg'
    npy_name = 'vgg64_bp.npy'
    accum_name = 'vgg64_bp_combined'

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

try:
    accum = np.load(accum_name + '.npy')
except:
    accum = conv1_filters

    accum = combine_filter(accum, conv2_filters, stride=1); print (np.shape(accum))

    accum = combine_filter(accum, conv3_filters, stride=2); print (np.shape(accum))
    accum = combine_filter(accum, conv4_filters, stride=2); print (np.shape(accum))

    accum = combine_filter(accum, conv5_filters, stride=4); print (np.shape(accum))
    accum = combine_filter(accum, conv6_filters, stride=4); print (np.shape(accum))

    accum = combine_filter(accum, conv7_filters, stride=8); print (np.shape(accum))
    accum = combine_filter(accum, conv8_filters, stride=8); print (np.shape(accum))

    accum = combine_filter(accum, conv9_filters,  stride=16); print (np.shape(accum))
    accum = combine_filter(accum, conv10_filters, stride=16); print (np.shape(accum))

    np.save(accum_name, accum)

#####

accum = accum[:, :, :, 0:16]
viz = []

for ii in range(16):
    upscale = scipy.misc.imresize(accum[:, :, :, ii], 2.)
    viz.append(upscale)
    
accum = np.stack(viz, axis=3)

#####

accum = accum - np.min(accum, axis=(2), keepdims=True)
accum = accum / np.max(accum, axis=(0,1,2), keepdims=True) 

viz_filter_3_channels(jpg_name, accum)







