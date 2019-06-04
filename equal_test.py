
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from conv import conv

#####

img = range(25)
img = np.reshape(img, (5, 5))

#####

f1 = np.array([[1.,0.,1.],[2.,0.,2.],[1.,0.,1.]])
f1 = np.reshape(f1, (3, 3))

f2 = np.array([[0.,1.,2.],[0.,2.,4.],[0.,1.,2.]])
f2 = np.reshape(f1, (3, 3))

#####

o1 = np.zeros(shape=(3, 3))

o1[0][0] = np.sum(img[0:3, 0:3] * f1) 
o1[0][1] = np.sum(img[0:3, 1:4] * f1) 
o1[0][2] = np.sum(img[0:3, 2:5] * f1) 

o1[1][0] = np.sum(img[1:4, 0:3] * f1) 
o1[1][1] = np.sum(img[1:4, 1:4] * f1) 
o1[1][2] = np.sum(img[1:4, 2:5] * f1) 

o1[2][0] = np.sum(img[2:5, 0:3] * f1) 
o1[2][1] = np.sum(img[2:5, 1:4] * f1) 
o1[2][2] = np.sum(img[2:5, 2:5] * f1) 

o2 = np.sum(o1 * f2)
print (o2)

#####

f3 = np.zeros(shape=(5, 5))

f3[0:3, 0:3] += f1 * f2[0][0]
f3[0:3, 1:4] += f1 * f2[0][1]
f3[0:3, 2:5] += f1 * f2[0][2]

f3[1:4, 0:3] += f1 * f2[1][0]
f3[1:4, 1:4] += f1 * f2[1][1]
f3[1:4, 2:5] += f1 * f2[1][2]

f3[2:5, 0:3] += f1 * f2[2][0]
f3[2:5, 1:4] += f1 * f2[2][1]
f3[2:5, 2:5] += f1 * f2[2][2]

o2 = np.sum(img * f3)
print (o2)





