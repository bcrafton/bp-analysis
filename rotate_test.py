
import numpy as np

# x = range(25)
# x = np.reshape(x, (5,5))

x = range(9)
x = np.reshape(x, (3,3))

# transpose not necessary for this.
x = np.flip(x, [0, 1])

print (x)
