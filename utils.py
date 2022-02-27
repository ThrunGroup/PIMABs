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


def get_logfile_name(args, bandit=True):
    suffix = '-b' if bandit else ''
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


