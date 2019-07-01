
import numpy as np
import scipy
import matplotlib.pyplot as plt

def factors(x):
    l = [] 
    for i in range(1, x + 1):
        if x % i == 0:
            l.append(i)
    
    mid = int(len(l) / 2)
    
    if (len(l) % 2 == 1):
        return [l[mid], l[mid]]
    else:
        return l[mid-1:mid+1]

def viz_filter(name, filters):
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
            
    plt.imsave(name, img, cmap='gray')

def viz_filter_3_channels(name, filters):
    fh, fw, fin, fout = np.shape(filters)
    filters = np.transpose(filters, (3, 0, 1, 2))
    assert(fin == 3)
    [nrows, ncols] = factors(fout)
    filters = np.reshape(filters, (nrows, ncols, fw, fh, fin))

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
            
    plt.imsave(name, img)
    
def viz_filter_1_channels(name, filters, resize=1.):
    fh, fw, fin, fout = np.shape(filters)
    filters = np.transpose(filters, (3, 0, 1, 2))
    assert(fin == 1)
    [nrows, ncols] = factors(fout)
    filters = np.reshape(filters, (nrows, ncols, fw, fh))

    for ii in range(nrows):
        for jj in range(ncols):
            if jj == 0:
                pad = np.pad(filters[ii][jj], [[3, 3], [3, 3]], mode='constant')
                row = pad
            else:
                pad = np.pad(filters[ii][jj], [[3, 3], [3, 3]], mode='constant')
                row = np.concatenate((row, pad), axis=1)
                
        if ii == 0:
            img = row
        else:
            img = np.concatenate((img, row), axis=0)
            
    img = scipy.misc.imresize(img, resize)
    plt.imsave(name, img, cmap='gray')
    
    
    
    
