
import numpy as np
import matplotlib.pyplot as plt
from combine_filter import combine_filter

f1 = np.array([[1.,0.,-1.],[2.,0.,-2.],[1.,0.,-1.]])
f2 = np.array([[0.,1.,2.],[0.,2.,4.],[0.,1.,2.]])

fout = combine_filter(f1, f2)
plt.imshow(fout)
plt.show()
