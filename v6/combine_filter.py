
import numpy as np

def combine_filter(f1, f2, stride=1):

    h1, w1, fin1, fout1 = np.shape(f1)
    h2, w2, fin2, fout2 = np.shape(f2)
    
    oh = h1 + 2*(h2 // 2) * stride
    ow = w1 + 2*(w2 // 2) * stride
    
    ofin = fin1
    ofout = fout2
    
    fout = np.zeros(shape=(oh, ow, ofin, ofout))
    
    for x in range(h2):
        for y in range(w2):
            for c in range(fout2):
                sh = x * stride ; eh = x * stride + h1
                sw = y * stride ; ew = y * stride + w1
                fout[sh:eh, sw:ew, :, c] += np.sum(f2[x, y, :, c] * f1, axis=3)

    return fout
    

