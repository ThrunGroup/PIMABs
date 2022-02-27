import os


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
