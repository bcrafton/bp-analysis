
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from conv import conv
from combine_filter import combine_filter

#####

img = np.random.uniform(low=0., high=1., size=(1, 64, 64, 3))

#####

f1 = np.load('vgg_stride_weights.npy').item()['conv1']
f2 = np.load('vgg_stride_weights.npy').item()['conv2']
f3 = np.load('vgg_stride_weights.npy').item()['conv3']
f4 = np.load('vgg_stride_weights.npy').item()['conv4']
f5 = np.load('vgg_stride_weights.npy').item()['conv5']
f6 = np.load('vgg_stride_weights.npy').item()['conv6']
f7 = np.load('vgg_stride_weights.npy').item()['conv7']
f8 = np.load('vgg_stride_weights.npy').item()['conv8']

#####

f1 = np.absolute(f1)
f2 = np.absolute(f2)
f3 = np.absolute(f3)
f4 = np.absolute(f4)
f5 = np.absolute(f5)
f6 = np.absolute(f6)
f7 = np.absolute(f7)
f8 = np.absolute(f8)

#####

f12       = combine_filter(f1,       f2, stride=1); print (np.shape(f12))
f123      = combine_filter(f12,      f3, stride=2); print (np.shape(f123))
f1234     = combine_filter(f123,     f4, stride=2); print (np.shape(f1234))
f12345    = combine_filter(f1234,    f5, stride=4); print (np.shape(f12345))
f123456   = combine_filter(f12345,   f6, stride=4); print (np.shape(f123456))
f1234567  = combine_filter(f123456,  f7, stride=8); print (np.shape(f1234567))
f12345678 = combine_filter(f1234567, f8, stride=8); print (np.shape(f12345678))

#####

f12345678 = f12345678 / np.max(f12345678)

_, _, fin, fout = np.shape(f12345678)

for ii in range(fin):
    for jj in range(fout):
        print (ii, jj)
        plt.imsave('./imgs/%d_%d.jpg' % (ii, jj), f12345678[:, :, ii, jj])
        
        


