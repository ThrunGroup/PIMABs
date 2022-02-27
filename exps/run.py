'''
This script contains scaffolding to run many experiments, listed in a config
file such as auto_exp_config.py.

This script will parse each line (= exp configuration) from the config file and
run the corresponding experiment. It can also run many experiments in parallel
by using the pool.apply_async calls instead of the explicit run_exp calls.
'''

import cProfile
import importlib
import multiprocessing as mp
import traceback
import os
import numpy as np
import copy  # Do not delete, used when doing things asynchronously below

import sys
sys.path.append('..')
from utils import get_args, get_logfile_name
from matching_pursuit import matching_pursuit, b_matching_pursuit


def remap_args(args, exp):
    '''
    Parses a config line (as a list) into an args variable (a Namespace).
    '''
    args.N = exp[1]
    args.d = exp[2]
    args.seed = exp[3]
    args.p = exp[4]
    args.batch_size = exp[5]
    args.max_iter = exp[6]
    args.tol = exp[7]
    args.residue = exp[8]
    return args

def run_exp(args, method_name, prof_name):
    '''
    Runs an experiment with the given parameters, and writes the results to the
    files.

    Note that the approach below to profiling is undocumented.
    See https://stackoverflow.com/questions/1584425/return-value-while-using-cprofile
    '''
    np.random.seed(args.seed)
    data = np.random.rand(args.N + 1, args.d)  # + 1 to generate the signal
    signal = data[0]
    atoms = data[1:] / np.linalg.norm(data[1:], axis=1).reshape(-1, 1)
    prof = cProfile.Profile()
    _ = prof.runcall(method_name, input_signal=signal, atoms=atoms, args=args)

    # Change the leading subdir from logs/ to profiles/
    prof_name = os.path.split(prof_name)
    prof_name = os.path.join('profiles', *prof_name[1:])
    prof.dump_stats(prof_name)


def main():
    '''
    Run all the experiments in the experiments lists specified by the -e
    argument, and write the final results (including logstrings) to files. Can
    run multiple experiments in parallel by using the pool.apply_async calls
    below instead of the explicit run_exp calls.
    '''
    args = get_args()  # Uses default values for now as placeholder to instantiate args
    imported_config = importlib.import_module(args.exp_config.strip('.py'))
    pool = mp.Pool()
    for exp in imported_config.experiments:
        args = remap_args(args, exp)
        logfile = os.path.join('..', get_logfile_name(args, bandit=False))
        if os.path.exists(logfile) and not args.force:
            print("Warning: already have data for experiment", logfile)
            continue
        else:
            print("Running exp:", logfile)

        '''
        WARNING: The apply_async calls below are NOT threadsafe. In particular,
        strings in python are lists, which means they are passed by reference.
        This means that if a NEW thread gets the SAME reference as the other
        threads, and updates the object, the OLD thread will write to the wrong
        file. Therefore, whenever using multiprocessing, need to copy.deepcopy()
        all the arguments. Don't need to do this for the explicit run_exp calls
        though since those references are used appropriately (executed
        sequentially)
        '''
        try:
            if exp[0] == 'MP':
                # pool.apply_async(run_exp, args=(copy.deepcopy(args), matching_pursuit, copy.deepcopy(logfile)+'-profile'))  # Copy inline to copy OTF
                run_exp(args, matching_pursuit, logfile + '-profile')
            elif exp[0] == 'B_MP':
                # pool.apply_async(run_exp, args=(copy.deepcopy(args), b_matching_pursuit, copy.deepcopy(logfile)+'-profile-b'))  # Copy inline to copy OTF
                run_exp(args, b_matching_pursuit, logfile + '-profile-b')
            else:
                raise Exception('Invalid algorithm specified')
        except Exception as _e:
            print(traceback.format_exc())

    pool.close()
    pool.join()
    print("Finished")


if __name__ == "__main__":
    main()
