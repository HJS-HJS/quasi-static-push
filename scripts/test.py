import ctypes
from ctypes import CDLL, POINTER
from ctypes import c_size_t, c_double

import numpy as np

mylib = ctypes.cdll.LoadLibrary('./libpath50.so')

M = np.ones((5,5))
q = np.ones(5)

result = mylib.pathlcp(M, q)
print(result)
