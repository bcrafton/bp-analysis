
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from conv import conv
from combine_filter import combine_filter
from viz_filter import viz_filter
from viz_filter import viz_filter_3_channels
from viz_filter import viz_filter_1_channels

#####

filters_name = 'bp'

if filters_name == 'lel':
    jpg_name = 'lel.jpg'
    npy_name = 'vgg64_lel.npy'
    
elif filters_name == 'bp':
    jpg_name = 'bp.jpg'
    npy_name = 'vgg64_bp.npy'

else:
    assert(False)

f1 = np.load(npy_name).item()['conv1']
f2 = np.load(npy_name).item()['conv2']
f3 = np.load(npy_name).item()['conv3']
f4 = np.load(npy_name).item()['conv4']
f5 = np.load(npy_name).item()['conv5']
f6 = np.load(npy_name).item()['conv6']
f7 = np.load(npy_name).item()['conv7']
f8 = np.load(npy_name).item()['conv8']

#####

f12       = combine_filter(f1,       f2, stride=1); print (np.shape(f12))
f123      = combine_filter(f12,      f3, stride=2); print (np.shape(f123))
f1234     = combine_filter(f123,     f4, stride=2); print (np.shape(f1234))
f12345    = combine_filter(f1234,    f5, stride=4); print (np.shape(f12345))
f123456   = combine_filter(f12345,   f6, stride=4); print (np.shape(f123456))
f1234567  = combine_filter(f123456,  f7, stride=8); print (np.shape(f1234567))
f12345678 = combine_filter(f1234567, f8, stride=8); print (np.shape(f12345678))

'''
'''

'''
f78       = combine_filter(f7, f8,       stride=1); print (np.shape(f78))
f678      = combine_filter(f6, f78,      stride=2); print (np.shape(f678))
f5678     = combine_filter(f5, f678,     stride=1); print (np.shape(f5678))
f45678    = combine_filter(f4, f5678,    stride=2); print (np.shape(f45678))
f345678   = combine_filter(f3, f45678,   stride=1); print (np.shape(f345678))
f2345678  = combine_filter(f2, f345678,  stride=2); print (np.shape(f2345678))
f12345678 = combine_filter(f1, f2345678, stride=1); print (np.shape(f12345678))
'''

#####

filters = f12345678

filters = filters - np.min(filters, axis=3, keepdims=True)
filters = filters / np.max(filters, axis=3, keepdims=True) 

viz_filter_1_channels(jpg_name, filters)







