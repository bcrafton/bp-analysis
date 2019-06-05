
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from conv import conv
from combine_filter import combine_filter

#####

img = range(25)
img = np.reshape(img, (1, 5, 5, 1))

f1 = np.array([[1.,0.,1.],[2.,0.,2.],[1.,0.,1.]])
f2 = np.array([[0.,1.,2.],[0.,2.,4.],[0.,1.,2.]])

#####

f = combine_filter(f1, f2)
f = np.reshape(f, (5, 5, 1, 1))

out = conv(img,  f, [1,1], 'valid')
print (out)

#####

f1 = np.reshape(f1, (3, 3, 1, 1))
f2 = np.reshape(f2, (3, 3, 1, 1))

out1 = conv(img,  f1, [1,1], 'valid')
out2 = conv(out1, f2, [1,1], 'valid')
print (out2)

#####

# oh shit we messed something up !!!
