
import numpy as np

def combine_filter(f1, f2):

    h1, w1, fin1, fout1 = np.shape(f1)
    h2, w2, fin2, fout2 = np.shape(f2)
    
    oh = h1 + 2 * (h1 // 2)
    ow = w1 + 2 * (w1 // 2)
    ofin = fin1
    ofout = fout2
    fout = np.zeros(shape=(oh, ow, ofin, ofout))
    
    for x in range(h2):
        for y in range(w2):
            for c1 in range(fin2):
                for c2 in range(fout2):
                    sh = x ; eh = x + h1
                    sw = y ; ew = y + w1
                    fout[sh:eh, sw:ew, :, c2] = fout[sh:eh, sw:ew, :, c2] + f2[x][y][c1][c2] * f1[:, :, :, c1]

    return fout
    

