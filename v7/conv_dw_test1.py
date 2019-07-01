
import numpy as np
import sympy

x1, x2, x3, f1, f2, f3 = sympy.symbols('x1 x2 x3 f1 f2 f3')

x = np.array([x1, x2, x3])
f = np.array([f1, f2, f3])

print (x * f)

#####

x111, x112, x113, \
x121, x122, x123, \
x131, x132, x133, \
x211, x212, x213, \
x221, x222, x223, \
x231, x232, x233, \
x311, x312, x313, \
x321, x322, x323, \
x331, x332, x333 = sympy.symbols(
"x111 x112 x113 \
x121 x122 x123  \
x131 x132 x133  \
x211 x212 x213  \
x221 x222 x223  \
x231 x232 x233  \
x311 x312 x313  \
x321 x322 x323  \
x331 x332 x333"
)



