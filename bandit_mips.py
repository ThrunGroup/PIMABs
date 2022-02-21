import numpy as np
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Solve MIPS problem")
    parser.add_argument("--get_ground_truth", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    return parser.parse_args()

def load_data(fname):
    return np.load(fname)

def bandit_solve(root, data):
    # Each point in data consists of an "arm"
    # Each pulling an arm corresponds to calculating one coordinate difference

    N = len(data)
    means = np.zeros(N)
    ucbs = np.zeros(N)
    lcbs = np.zeros(N)
    pull_counts = np.zeros(N)

    # Identify arms to pull

    ### This many not work because there aren't any ROUNDS. Unless you use projection or do many ITERATIONS of MIPS
    ### Need a SEQUENCE of MAB problems


def main(args):
    cwd = os.getcwd()
    fname = os.path.join(cwd, "data", "data.npy")
    data = load_data(fname)

    if args.debug:
        print(data)

    root = data[0]

    if args.get_ground_truth:
        mip = -float("inf")
        best_p = -1
        for p_idx, p in enumerate(data):
            if p_idx == 0:
                continue

            if args.debug:
                print(p_idx, p.dot(root))

            if p.dot(root) > mip:
                mip = p.dot(root)
                best_p = p_idx

        return best_p, mip


if __name__ == '__main__':
    args = get_args()
    print(main(args))