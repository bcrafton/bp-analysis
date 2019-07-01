
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from conv import conv

from combine_filter import combine_filter
from combine_filter import combine_filter_dw

from viz_filter import viz_filter
from viz_filter import viz_filter_3_channels
from viz_filter import viz_filter_1_channels

#####

filters_name = 'lel'

if filters_name == 'lel':
    jpg_name = 'lel.jpg'
    npy_name = 'mobile64_lel.npy'
    
elif filters_name == 'bp':
    jpg_name = 'bp.jpg'
    npy_name = 'mobile64_bp.npy'

else:
    assert(False)


weights = np.load(npy_name).item()

conv1 = weights['block1_conv']

conv_dw_2 = weights['block2_conv_block_dw_conv_dw']
conv_pw_2 = weights['block2_conv_block_pw_conv']

conv_dw_3 = weights['block3_conv_block_dw_conv_dw']
conv_pw_3 = weights['block3_conv_block_pw_conv']

conv_dw_4 = weights['block4_conv_block_dw_conv_dw']
conv_pw_4 = weights['block4_conv_block_pw_conv']

conv_dw_5 = weights['block5_conv_block_dw_conv_dw']
conv_pw_5 = weights['block5_conv_block_pw_conv']

conv_dw_6 = weights['block6_conv_block_dw_conv_dw']
conv_pw_6 = weights['block6_conv_block_pw_conv']

conv_dw_7 = weights['block7_conv_block_dw_conv_dw']
conv_pw_7 = weights['block7_conv_block_pw_conv']

conv_dw_8 = weights['block8_conv_block_dw_conv_dw']
conv_pw_8 = weights['block8_conv_block_pw_conv']

conv_dw_9 = weights['block9_conv_block_dw_conv_dw']
conv_pw_9 = weights['block9_conv_block_pw_conv']

conv_dw_10 = weights['block10_conv_block_dw_conv_dw']
conv_pw_10 = weights['block10_conv_block_pw_conv']

'''
# should exist but we messed up in MobileNet64.
conv_dw_11 = weights['block11_conv_block_dw_conv_dw']
conv_pw_11 = weights['block11_conv_block_pw_conv']
'''

#####

'''
f12       = combine_filter_dw(f1,       f2, stride=1); print (np.shape(f12))
f123      = combine_filter   (f12,      f3, stride=2); print (np.shape(f123))
f1234     = combine_filter_dw(f123,     f4, stride=2); print (np.shape(f1234))
f12345    = combine_filter   (f1234,    f5, stride=4); print (np.shape(f12345))
f123456   = combine_filter_dw(f12345,   f6, stride=4); print (np.shape(f123456))
f1234567  = combine_filter   (f123456,  f7, stride=8); print (np.shape(f1234567))
f12345678 = combine_filter_dw(f1234567, f8, stride=8); print (np.shape(f12345678))
'''

#####

'''
l1_1 = ConvBlock(input_shape=[batch_size, 64, 64, 3], filter_shape=[3, 3, 3, 32], strides=[1,1,1,1], init=args.init, name='block1')
l1_2 = LELConv(input_shape=[batch_size, 64, 64, 32], pool_shape=[1,8,8,1], num_classes=1000, name='block1_fb')

l2 = MobileBlock(input_shape=[batch_size, 64, 64, 32],  filter_shape=[32, 64],   strides=[1,2,2,1], init=args.init, pool_shape=[1,8,8,1], num_classes=1000, name='block2')
l3 = MobileBlock(input_shape=[batch_size, 32, 32, 64],  filter_shape=[64, 128],  strides=[1,1,1,1], init=args.init, pool_shape=[1,8,8,1], num_classes=1000, name='block3')

l4 = MobileBlock(input_shape=[batch_size, 32, 32, 128], filter_shape=[128, 256], strides=[1,2,2,1], init=args.init, pool_shape=[1,4,4,1], num_classes=1000, name='block4')
l5 = MobileBlock(input_shape=[batch_size, 16, 16, 256], filter_shape=[256, 512], strides=[1,1,1,1], init=args.init, pool_shape=[1,4,4,1], num_classes=1000, name='block5')

l6 = MobileBlock(input_shape=[batch_size, 16, 16, 512], filter_shape=[512, 512], strides=[1,2,2,1], init=args.init, pool_shape=[1,2,2,1], num_classes=1000, name='block6')
l7 = MobileBlock(input_shape=[batch_size, 8, 8, 512], filter_shape=[512, 512], strides=[1,1,1,1], init=args.init, pool_shape=[1,2,2,1], num_classes=1000, name='block7')

l8 = MobileBlock(input_shape=[batch_size, 8, 8, 512], filter_shape=[512, 512], strides=[1,1,1,1], init=args.init, pool_shape=[1,2,2,1], num_classes=1000, name='block8')
l9 = MobileBlock(input_shape=[batch_size, 8, 8, 512], filter_shape=[512, 512], strides=[1,1,1,1], init=args.init, pool_shape=[1,2,2,1], num_classes=1000, name='block9')

l10 = MobileBlock(input_shape=[batch_size, 8, 8, 512],  filter_shape=[512, 1024],  strides=[1,2,2,1], init=args.init, pool_shape=[1,2,2,1], num_classes=1000, name='block10')
l11 = MobileBlock(input_shape=[batch_size, 4, 4, 1024], filter_shape=[1024, 1024], strides=[1,1,1,1], init=args.init, pool_shape=[1,4,4,1], num_classes=1000, name='block11')

# BUG! double named l11.
l11 = AvgPool(size=[batch_size, 4, 4, 1024], ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")
l12 = ConvToFullyConnected(input_shape=[1, 1, 1024])
l13 = FullyConnected(input_shape=1024, size=1000, init=args.init, name="fc1")
'''

#####

accum = conv1

accum = combine_filter_dw(accum, conv_dw_2, stride=1); print (np.shape(accum))
accum = combine_filter   (accum, conv_pw_2, stride=2); print (np.shape(accum))
accum = combine_filter_dw(accum, conv_dw_3, stride=2); print (np.shape(accum))
accum = combine_filter   (accum, conv_pw_3, stride=2); print (np.shape(accum))

accum = combine_filter_dw(accum, conv_dw_4, stride=2); print (np.shape(accum))
accum = combine_filter   (accum, conv_pw_4, stride=4); print (np.shape(accum))
accum = combine_filter_dw(accum, conv_dw_5, stride=4); print (np.shape(accum))
accum = combine_filter   (accum, conv_pw_5, stride=4); print (np.shape(accum))

accum = combine_filter_dw(accum, conv_dw_6, stride=4); print (np.shape(accum))
accum = combine_filter   (accum, conv_pw_6, stride=8); print (np.shape(accum))
accum = combine_filter_dw(accum, conv_dw_7, stride=8); print (np.shape(accum))
accum = combine_filter   (accum, conv_pw_7, stride=8); print (np.shape(accum))

accum = combine_filter_dw(accum, conv_dw_8, stride=8); print (np.shape(accum))
accum = combine_filter   (accum, conv_pw_8, stride=8); print (np.shape(accum))
accum = combine_filter_dw(accum, conv_dw_9, stride=8); print (np.shape(accum))
accum = combine_filter   (accum, conv_pw_9, stride=8); print (np.shape(accum))

accum = combine_filter_dw(accum, conv_dw_10, stride=8);  print (np.shape(accum))
accum = combine_filter   (accum, conv_pw_10, stride=16); print (np.shape(accum))
'''
accum = combine_filter_dw(accum, conv_dw_11, stride=16); print (np.shape(accum))
accum = combine_filter   (accum, conv_pw_11, stride=16); print (np.shape(accum))
'''

#####
'''
accum = accum[:, :, :, 0:64]

# think about what this is doing.

# subtract min:
# subtracting min along rgb axis will give black = good

# divide max:
# dividing max along this axis will make them all equal = gray = bad.

accum = accum - np.min(accum, axis=(2), keepdims=True)

# pretty, but not accurate.
# accum = accum / np.max(accum, axis=(2), keepdims=True) 

accum = accum / np.max(accum, axis=(0,1,2), keepdims=True) 

viz_filter_3_channels(jpg_name, accum)

# print (np.average(accum))
'''
#####

x = np.load('imagenet64_example.npy')

x = np.reshape(x[0], (1, 64, 64, 3))
print (np.shape(x))

out = conv(x, accum, [16,16], 'same')
print (np.shape(out))

np.save('act', out)

#####







