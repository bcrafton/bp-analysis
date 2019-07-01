
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from conv import conv
from combine_filter import combine_filter
from viz_filter import viz_filter
from viz_filter import viz_filter_3_channels
from viz_filter import viz_filter_1_channels

#####

lel = 0
dim3 = 0

if lel:
    jpg_name = 'lel.jpg'
    npy_name = 'mobile64_lel.npy'
else:
    jpg_name = 'bp.jpg'
    npy_name = 'mobile64_bp.npy'

f1 = np.load(npy_name).item()['block2_conv_block_dw_conv_dw']
f2 = np.load(npy_name).item()['block2_conv_block_pw_conv']
f3 = np.load(npy_name).item()['block3_conv_block_dw_conv_dw']

#####
# this def aint right .


f1 = np.transpose(f1, (0, 1, 3, 2))
f12 = combine_filter(f1, f2, stride=2)
f12 = f12[:, :, :, 0]
f12 = np.reshape(f12, (3, 3, 1, 1))

f3 = np.transpose(f3, (0, 1, 3, 2))

f123 = combine_filter(f12, f3, stride=2)

#####

filters = f123

# filters = filters - np.min(filters, axis=(3), keepdims=True)
filters = filters / np.max(filters, axis=(3), keepdims=True) 

viz_filter_1_channels(jpg_name, filters, resize=8.)

#####





