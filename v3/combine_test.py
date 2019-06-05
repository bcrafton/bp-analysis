
import numpy as np
import matplotlib.pyplot as plt
from combine_filter import combine_filter

f1 = np.random.uniform(size=(3, 3, 3, 32))
f2 = np.random.uniform(size=(3, 3, 32, 64))

fout = combine_filter(f1, f2, stride=2)
print (np.shape(fout))
