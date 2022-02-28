'''
Convenience code to automatically generate a list of experiments to run.
Default output is to auto_exp_config.py.
'''

import itertools

import sys
sys.path.append('..')
from utils import write_exp

def main():
    algos = ["B_MP"]#], "MP"]
    Ns = [783]
    ds = [
        100,
        200,
        400,
        800,
        1600,
        3200,
        6400,
        12800,
        25600,
        51200,
    ]
    seeds = range(10)
    ps = [1e-2]  # This is unused by the experiments and recomputed in-method
    batch_sizes = [10]  # This is unused by the experiments and recomputed in-method
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
