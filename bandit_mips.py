import numpy as np
import os
import argparse
import sys



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--get_ground_truth", type=bool, default=False)
    return parser.parse_args()


def load_data(fname):
    return np.load(fname)

def main():
    cwd = os.getcwd()
    fname = os.path.join(cwd, "data.npy")
    data = load_data(fname)

    root = data[0]

    if get_ground_truth:
        ground_truth

if __name__ == '__main__':
    main(sys.argv[1:]