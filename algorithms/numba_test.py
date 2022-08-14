from numba import njit
import numpy as np

@njit
def foo():
    d = dict()
    k = {1: np.arange(1), 2: np.arange(2)}
    # The following tells the compiler what the key type and the
    # value
    # type are for `d`.
    d[3] = np.arange(3)
    d[5] = np.arange(5)
    return d, k

d, k = foo()
print(d)    # {3: [0 1 2], 5: [0 1 2 3 4]}
print(k)    # {1: [0], 2: [0 1]}