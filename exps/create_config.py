'''
Convenience code to automatically generate a list of experiments to run.
Default output is to auto_exp_config.py.
'''

import itertools

import sys
sys.path.append('..')
from utils import write_exp

def main():
    algos = ["B_MP", "MP"]
    Ns = [100]
    ds = [
        1000,
        3000,
        10000,
        30000,
        100000,
        300000,
        1000000,
        #3000000,
        #10000000,
    ]
    seeds = range(10)
    ps = [1e-2]  # This is unused by the experiments and recomputed in-method
    batch_sizes = [100]  # This is unused by the experiments and recomputed in-method
    max_iters = [1]
    tols = [1e-2]
    residues = [1e-2]

    exps = itertools.product(algos, Ns, ds, seeds, ps, batch_sizes, max_iters, tols, residues)
    with open('auto_exp_config.py', 'w+') as fout:
        fout.write("experiments = [\n")
        for e in exps:
            exp = write_exp(*e)
            fout.write(exp)
        fout.write("]")


if __name__ == "__main__":
    main()
