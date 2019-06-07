
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from conv import conv
from combine_filter import combine_filter

#####

def viz(name, filters):
    fh, fw, fin, fout = np.shape(filters)
    filters = filters.T
    assert(np.shape(filters) == (fout, fin, fw, fh))
    [nrows, ncols] = factors(fin * fout)
    filters = np.reshape(filters, (nrows, ncols, fw, fh))

    for ii in range(nrows):
        for jj in range(ncols):
            if jj == 0:
                row = filters[ii][jj]
            else:
                row = np.concatenate((row, filters[ii][jj]), axis=1)
                
        if ii == 0:
            img = row
        else:
            img = np.concatenate((img, row), axis=0)
            
    plt.imsave(name, img, cmap="gray")

#####

img = np.random.uniform(low=0., high=1., size=(1, 64, 64, 3))

#####

lel = 1
dim3 = 0

if lel:
    f1 = np.load('vgg64_lel.npy').item()['conv1']
    f2 = np.load('vgg64_lel.npy').item()['conv2']
    f3 = np.load('vgg64_lel.npy').item()['conv3']
    f4 = np.load('vgg64_lel.npy').item()['conv4']
    f5 = np.load('vgg64_lel.npy').item()['conv5']
    f6 = np.load('vgg64_lel.npy').item()['conv6']
    f7 = np.load('vgg64_lel.npy').item()['conv7']
    f8 = np.load('vgg64_lel.npy').item()['conv8']
else:
    f1 = np.load('vgg64_bp.npy').item()['conv1']
    f2 = np.load('vgg64_bp.npy').item()['conv2']
    f3 = np.load('vgg64_bp.npy').item()['conv3']
    f4 = np.load('vgg64_bp.npy').item()['conv4']
    f5 = np.load('vgg64_bp.npy').item()['conv5']
    f6 = np.load('vgg64_bp.npy').item()['conv6']
    f7 = np.load('vgg64_bp.npy').item()['conv7']
    f8 = np.load('vgg64_bp.npy').item()['conv8']

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

if dim3:
    for ii in range(fout):
        print (ii)
        plt.imsave('./imgs/%d.jpg' % (ii), filters[:, :, :, ii])

else:
    for ii in range(fin):
        for jj in range(fout):
            print (ii, jj)
            plt.imsave('./imgs/%d_%d.jpg' % (ii, jj), filters[:, :, ii, jj])




