
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

if args.alg == 'lel':
    jpg_name = 'lel.jpg'
    npy_name = 'mobile64_lel.npy'
    accum_name = 'mobile64_lel_combined'
    
elif args.alg == 'bp':
    jpg_name = 'bp.jpg'
    npy_name = 'mobile64_bp.npy'
    accum_name = 'mobile64_bp_combined'

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

conv11_dw_filters = weights['block11_conv_block_dw_conv_dw']
conv11_pw_filters = weights['block11_conv_block_pw_conv']

#####

accum = conv1_filters

accum = combine_filter_dw(accum, conv2_dw_filters, stride=1); print (np.shape(accum))
accum = combine_filter   (accum, conv2_pw_filters, stride=1); print (np.shape(accum))
accum = combine_filter_dw(accum, conv3_dw_filters, stride=2); print (np.shape(accum))
accum = combine_filter   (accum, conv3_pw_filters, stride=1); print (np.shape(accum))

accum = combine_filter_dw(accum, conv4_dw_filters, stride=2); print (np.shape(accum))
accum = combine_filter   (accum, conv4_pw_filters, stride=1); print (np.shape(accum))
accum = combine_filter_dw(accum, conv5_dw_filters, stride=4); print (np.shape(accum))
accum = combine_filter   (accum, conv5_pw_filters, stride=1); print (np.shape(accum))

accum = combine_filter_dw(accum, conv6_dw_filters, stride=4); print (np.shape(accum))
accum = combine_filter   (accum, conv6_pw_filters, stride=1); print (np.shape(accum))
accum = combine_filter_dw(accum, conv7_dw_filters, stride=8); print (np.shape(accum))
accum = combine_filter   (accum, conv7_pw_filters, stride=1); print (np.shape(accum))

accum = combine_filter_dw(accum, conv8_dw_filters, stride=8); print (np.shape(accum))
accum = combine_filter   (accum, conv8_pw_filters, stride=1); print (np.shape(accum))
accum = combine_filter_dw(accum, conv9_dw_filters, stride=8); print (np.shape(accum))
accum = combine_filter   (accum, conv9_pw_filters, stride=1); print (np.shape(accum))

accum = combine_filter_dw(accum, conv10_dw_filters, stride=8);  print (np.shape(accum))
accum = combine_filter   (accum, conv10_pw_filters, stride=1);  print (np.shape(accum))
accum = combine_filter_dw(accum, conv11_dw_filters, stride=16); print (np.shape(accum))
accum = combine_filter   (accum, conv11_pw_filters, stride=1);  print (np.shape(accum))

#####
np.save(accum_name, accum)

accum = accum[:, :, :, 0:128]

# think about what this is doing.

# subtract min:
# subtracting min along rgb axis will give black = good

# divide max:
# dividing max along this axis will make them all equal = gray = bad.

accum = accum - np.min(accum, axis=(2), keepdims=True)

# pretty, but not accurate.
# accum = accum / np.max(accum, axis=(2), keepdims=True) 

accum = accum / np.max(accum, axis=(0,1,2), keepdims=True) 

print (np.average(accum))

viz_filter_3_channels(jpg_name, accum)
#####







