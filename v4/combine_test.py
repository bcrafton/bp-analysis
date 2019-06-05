
import numpy as np
import matplotlib.pyplot as plt
from combine_filter import combine_filter

f1 = np.random.uniform(size=(5, 5, 3, 32))
f2 = np.random.uniform(size=(5, 5, 32, 64))

fout = combine_filter(f1, f2, stride=2)
print (np.shape(fout))
print (fout[:, :, 0, 0])
