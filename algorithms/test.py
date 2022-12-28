import numpy as np

N = 100
arr = np.arange(N)
k = 30
m = N // k

for i in range(k):
    if i == k - 1:
        print(arr[i * m:])
    else:
        print(arr[i * m : (i + 1) * m])