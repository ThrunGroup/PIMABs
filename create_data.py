import numpy as np
import os

def create_data(N=1e4, d=1e4, fname=os.path.join("data", "data.npy")):
    data = np.random.rand(N, d)
    np.save(fname, data)


if __name__ == '__main__':
    N = 10
    d = 10
    create_data(N=N, d=d)
    print(N)