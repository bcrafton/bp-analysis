
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from conv import conv
from combine_filter import combine_filter

#####

vgg_weights = np.load('vgg_weights.npy').item()

f1 = vgg_weights['conv1']
f2 = vgg_weights['conv2']
f3 = vgg_weights['conv3']
f4 = vgg_weights['conv4']
f5 = vgg_weights['conv5']
f6 = vgg_weights['conv6']
f7 = vgg_weights['conv7']
f8 = vgg_weights['conv8']
f9 = vgg_weights['conv9']
f10 = vgg_weights['conv10']
f11 = vgg_weights['conv11']
f12 = vgg_weights['conv12']
f13 = vgg_weights['conv13']

#####

c2  = combine_filter(f1, f2,   stride=1);  print (np.shape(c2))
c3  = combine_filter(c2, f3,   stride=2);  print (np.shape(c3))
c4  = combine_filter(c3, f4,   stride=2);  print (np.shape(c4))
c5  = combine_filter(c4, f5,   stride=4);  print (np.shape(c5))
c6  = combine_filter(c5, f6,   stride=4);  print (np.shape(c6))
c7  = combine_filter(c6, f7,   stride=4);  print (np.shape(c7))
c8  = combine_filter(c7, f8,   stride=8);  print (np.shape(c8))
c9  = combine_filter(c8, f9,   stride=8);  print (np.shape(c9))
c10 = combine_filter(c9, f10,  stride=8);  print (np.shape(c10))
c11 = combine_filter(c10, f11, stride=16); print (np.shape(c11))
c12 = combine_filter(c11, f12, stride=16); print (np.shape(c12))
c13 = combine_filter(c12, f13, stride=16); print (np.shape(c13))

#####

filters = c13

filters = filters / np.max(filters)
_, _, fin, fout = np.shape(filters)

'''
for ii in range(fin):
    for jj in range(fout):
        print (ii, jj)
        plt.imsave('./imgs/%d_%d.jpg' % (ii, jj), filters[:, :, ii, jj])
'''
#'''
for ii in range(fout):
    print (ii)
    plt.imsave('./imgs/%d.jpg' % (ii), filters[:, :, :, ii])
#'''






