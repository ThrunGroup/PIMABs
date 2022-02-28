import os
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Solve matching pursuit problems')
    parser.add_argument('-N', default=100, type=int)
    parser.add_argument('-d', default=1000, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--p', type=float)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--max_iter', default=10, type=int)
    parser.add_argument('--tol', default=1e-2, type=float)
    parser.add_argument('--residue', default=1e-3, type=float)
    parser.add_argument('-e', '--exp_config', help='Experiment configuration file to use', required=False)
    parser.add_argument('-f', '--force', help='Recompute Experiments', action='store_true')
    return parser.parse_args()


def remap_args(args, exp):
    '''
    Parses a config line (as a list) into an args variable (a Namespace).
    '''
    args.N = exp[1]
    args.d = exp[2]
    args.seed = exp[3]
    args.p = None  # To compute on the fly
    args.batch_size = exp[5]
    args.max_iter = exp[6]
    args.tol = exp[7]
    args.residue = exp[8]
    return args


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


def get_logfile_name(args, bandit=True):
    suffix = '-bmp' if bandit else ''
    return os.path.join('logs',
                'N-' + str(args.N) +
                '-d-' + str(args.d) +
                '-s-' + str(args.seed) +
                '-p-' + str(args.p) +
                '-b-' + str(args.batch_size) +
                '-m-' + str(args.max_iter) +
                '-r-' + str(args.residue) +
                '-t-' + str(args.tol) +
                        suffix
                        )


