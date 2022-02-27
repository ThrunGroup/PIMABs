'''
Convenience code to automatically generate a list of experiments to run.
Default output is to auto_exp_config.py.
'''

import itertools

def q(str_):
    """
    Puts quotes around a string
    """
    return '"' + str_ + '"'

def write_exp(algo, N, d, seed, p, batch_size, max_iter, tol, residue):
    '''
    Takes the experiment variables and outputs a string description
    '''

    # Can compute p and batch_size here if necessary
    return "\t[" + ', '.join(map(str, [q(algo), N, d, seed, p, batch_size, max_iter, tol, residue])) + "],\n"

def main():
    algos = ["B_MP", "MP"]
    Ns = [100]
    ds = [100, 1000, 1000]
    seeds = range(10)
    ps = [1e-2]
    batch_sizes = [1000]
    max_iters = [10]
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
