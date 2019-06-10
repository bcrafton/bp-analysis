
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from conv import conv
from combine_filter import combine_filter
from viz_filter import viz_filter
from viz_filter import viz_filter_3_channels

#####

lel = 1
dim3 = 0

if lel:
    jpg_name = 'lel.jpg'
    # npy_name = 'vgg64_lel.npy'
    npy_name = 'vgg64_lel_1x1.npy'
else:
    jpg_name = 'bp.jpg'
    # npy_name = 'vgg64_bp.npy'
    npy_name = 'vgg64_bp_5_epoch.npy'

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

#####

filters = f12345678

filters = filters / np.max(filters)
_, _, fin, fout = np.shape(filters)

viz_filter_3_channels(jpg_name, filters)

'''
for ii in range(fin):
    for jj in range(fout):
        print (np.linalg.matrix_rank(filters[:, :, ii, jj]))
'''

'''
if dim3:
    for ii in range(fout):
        print (ii)
        plt.imsave('./imgs/%d.jpg' % (ii), filters[:, :, :, ii])

else:
    for ii in range(fin):
        for jj in range(fout):
            print (ii, jj)
            plt.imsave('./imgs/%d_%d.jpg' % (ii, jj), filters[:, :, ii, jj])
'''


