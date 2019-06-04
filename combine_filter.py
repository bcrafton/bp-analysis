
import numpy as np

def combine_filter(f1, f2):

    h1, w1 = np.shape(f1)
    h2, w2 = np.shape(f2)
    
    oh = h1 + 2 * (h1 // 2)
    ow = w1 + 2 * (w1 // 2)
    fout = np.zeros(shape=(oh, ow))
    
    for ii in range(h2):
        for jj in range(w2):
            sh = ii
            sw = jj 
            eh = ii + h1
            ew = jj + w1
            fout[sh:eh, sw:ew] = fout[sh:eh, sw:ew] + f2[ii][jj] * f1
            
    return fout
