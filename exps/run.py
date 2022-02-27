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
import copy
import traceback
import os

from utils import get_args, get_logfile_name


def remap_args(args, exp):
    '''
    Parses a config line (as a list) into an args variable (a Namespace).
    '''
    args.N = exp[0]
    args.d = exp[1]
    args.seed = exp[2]
    args.p = exp[3]
    args.batch_size = exp[4]
    args.max_iter = exp[5]
    args.tol = exp[6]
    args.residue = exp[7]
    return args

def run_exp(args, method_name, medoids_fname, B_prof_fname, S_prof_fname):
    '''
    Runs an experiment with the given parameters, and writes the results to the
    files.

    Note that the approach below to profiling is undocumented.
    See https://stackoverflow.com/questions/1584425/return-value-while-using-cprofile
    '''
    prof = cProfile.Profile()
    _ = prof.runcall(method_name, args)
    prof.dump_stats(B_prof_fname)


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
            if exp[0] == 'naive_v1':
                # pool.apply_async(run_exp, args=(copy.deepcopy(args), naive_pam_v1.naive_build_and_swap, copy.deepcopy(medoids_fname), copy.deepcopy(B_prof_fname), copy.deepcopy(S_prof_fname))) # Copy inline to copy OTF
                run_exp(args, naive_pam_v1.naive_build_and_swap, medoids_fname, B_prof_fname, S_prof_fname)
            elif exp[0] == 'ucb':
                # pool.apply_async(run_exp, args=(copy.deepcopy(args), ucb_pam.UCB_build_and_swap, copy.deepcopy(medoids_fname), copy.deepcopy(B_prof_fname), copy.deepcopy(S_prof_fname)))
                run_exp(args, ucb_pam.UCB_build_and_swap, medoids_fname, B_prof_fname, S_prof_fname)
            elif exp[0] == 'clarans':
                # pool.apply_async(run_loss_exp, args=(copy.deepcopy(args), clarans.CLARANS_build_and_swap, copy.deepcopy(medoids_fname)))
                run_loss_exp(copy.deepcopy(args), clarans.CLARANS_build_and_swap, copy.deepcopy(medoids_fname))
            elif exp[0] == 'em_style':
                # pool.apply_async(run_loss_exp, args=(copy.deepcopy(args), em_style.EM_build_and_swap, copy.deepcopy(medoids_fname)))
                run_loss_exp(copy.deepcopy(args), em_style.EM_build_and_swap, copy.deepcopy(medoids_fname))
            else:
                raise Exception('Invalid algorithm specified')
        except Exception as _e:
            print(traceback.format_exc())

    pool.close()
    pool.join()
    print("Finished")


if __name__ == "__main__":
    main()
