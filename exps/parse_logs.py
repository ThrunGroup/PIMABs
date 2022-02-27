'''
Code to automatically parse the profiles produced from running experiments.
In particular, plots the scaling of B_MP vs. d for various dimension sizes d.
Used to demonstrate O(dlogd) scaling.
'''

import sys
import os
import pstats
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools
import snakevizcode

import sys
sys.path.append('..')
from utils import get_args, get_logfile_name, remap_args

# Possible line numbers for the empty_counter fn
# FN_NAME_1 = 'data_utils.py:129(empty_counter)'
# FN_NAME_2 = 'data_utils.py:141(empty_counter)'
# FN_NAME_3 = 'data_utils.py:142(empty_counter)'

def verify_correctness():
    '''
    Verifies that BanditPAM returns the same BUILD and SWAP medoid assignments
    as PAM, by parsing the logfiles.
    '''

    parent_dirs = [
        'logs',
    ]

    for parent_dir in parent_dirs:
        bmp_logfiles = [os.path.join(parent_dir, x) for x in os.listdir(parent_dir)
                        if os.path.isfile(os.path.join(parent_dir, x)) and x != '.DS_Store' and x[-4:] == '-bmp']

        for bmp_lfile in sorted(bmp_logfiles):
            mp_lfile = bmp_lfile.replace('-bmp', '')
            if not os.path.exists(mp_lfile):
                print("Warning: no naive experiment", mp_lfile)
            else:
                disagreement = False
                with open(bmp_lfile, 'r') as fin1:
                    with open(mp_lfile, 'r') as fin2:
                        # Skip the first lines since the costs may be different
                        bmp_l = fin1.readline().strip()
                        mp_l = fin2.readline().strip()

                        while bmp_l and mp_l:
                            bmp_l = fin1.readline().strip()
                            mp_l = fin2.readline().strip()
                            if bmp_l != mp_l:
                                disagreement = True

                if disagreement:
                    print("ERROR: Results for", bmp_lfile, mp_lfile, "disagree!!")
                else:
                    print("OK: Results for", bmp_lfile, mp_lfile, "agree")


def plot_slice_sns(dcalls_array, ds, costs, algo, seeds, take_log=True):
    '''
    Plots the number of coordinate-wise distance calls vs. d, for various algorithms and seeds

    :param: dcalls_array, an array of d, seed, calls
    '''

    if take_log:
        np_data = np.log10(dcalls_array)
        xs = np.log10(ds)
    else:
        np_data = dcalls_array
        xs = ds

    sns.set()
    sns.set_style('white')

    fig, ax = plt.subplots(figsize=(7, 7))

    # Make a dataframe with the relevant data, for plotting with seaborn
    d = {'d': xs}
    for seed_idx, seed in enumerate(seeds):
        d["seed_" + str(seed)] = np_data[:, seed_idx]
    df = pd.DataFrame(data=d)
    print(df)

    # Combine the different seeds into 1 column
    melt_df = df.melt('N', var_name='cols', value_name='vals')
    melt_df['N'] += np.random.randn(melt_df['N'].shape[0]) * 0.01 # Add jitter
    sns.scatterplot(x="N", y="vals", data = melt_df, ax = ax, alpha = 0.6)

    # Plot means and error bars
    bars = (1.96/(10**0.5)) * np.std(np_data[kN_idx, :, :], axis = 1) # Slice a specific k, get a 2D array
    means = np.mean(np_data[kN_idx, :, :], axis = 1)
    plt.errorbar(Nks_plot, means, yerr = bars, fmt = '+', capsize = 5, ecolor='black', elinewidth = 1.5, zorder = 100, mec='black', mew = 1.5, label="95% confidence interval")

    # Plot line of best fit
    sl, icpt, r_val, p_val, _ = sp.stats.linregress(Nks_plot, means)
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    plt.plot([x_min, x_max], [x_min * sl + icpt, x_max * sl + icpt], color='black', label='Linear fit, slope=%0.3f'%(sl))

    # Plot reference kN^2 line here in build step for PAM
    plt.plot([x_min, x_max], [np.log10(kN) + x_min * 2, np.log10(kN) + x_max * 2], color='red', label='$kn^2$ PAM scaling')
    # Plot reference N^2 line here in build step for PAM + FP1
    # (no dependence on k)
    plt.plot([x_min, x_max], [x_min * 2, x_max * 2], color='red', label='$kn^2$ PAM scaling')
    # weighted reference line for
    plt.plot([x_min, x_max], [(x_min) * 2, (x_max) * 2], color='#f781bf', linestyle=':', label='$n^2$ FastPAM1 scaling')
    plt.plot([x_min, x_max], [np.log10(kN) + (x_min) * 2, np.log10(kN) + (x_max) * 2], color='#4daf4a', linestyle='-.', label='$kn^2$ PAM scaling')

    print("Slope is:", sl)

    # Manually modify legend labels for prettiness
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles[::-1], labels[::-1], loc="upper left")


    plt.xlabel("$\log_{10}(n)$")
    plt.ylabel("$\log_{10}$(# of coordinate-wise distance computations per step)")

    # Modify these lines based on dataset
    plt.title("HOC4, $d =$ tree edit distance, $k = 2$")
    plt.show()
    plt.savefig('figures/HOC4-with-FP1.pdf')


def show_plots():
    '''
    A function which mines the number of distance calls for each experiment,
    from the dumped profiles. Creates a numpy array with the distance call
    counts.

    It does this by:
        - first, identifying the filenames where the experiment profiles and
            logfiles are stored (the logfile is used for the number of swap
            steps)
        - searching each profile (build and swap) for the number of distance
            calls
        - weighting the distance calls between the build step and swap step as
            necessary
    '''
    args = get_args()  # Uses default values for now as placeholder to instantiate args

    ### Decide which experiments to plot
    algos = ["B_MP"]  # , "MP"]
    Ns = [100]
    ds = [3000, 10000, 30000, 100000, 300000]
    dmap = {
        3000: 0,
        10000: 1,
        30000: 2,
        100000: 3,
        300000: 4,
    }
    seeds = range(10)
    ps = [1e-2]  # This is unused by the experiments and recomputed in-method
    batch_sizes = [1000]  # This is unused by the experiments and recomputed in-method
    max_iters = [10]
    tols = [1e-2]
    residues = [1e-2]
    exps = itertools.product(algos, Ns, ds, seeds, ps, batch_sizes, max_iters, tols, residues)

    ### Gather data
    dcalls_array = np.zeros((len(ds), len(seeds)))
    for exp in exps:
        args = remap_args(args, exp)
        logfile = os.path.join(get_logfile_name(args, bandit=True))
        if not os.path.exists(logfile):
            raise Exception('No logfile for experiment', logfile)

        with open(logfile, 'r') as fin:
            cost = fin.readline().strip('Total cost: ')
            if dcalls_array[dmap[args.d], args.seed] != 0:
                raise Exception('Rewriting data matrix on', logfile)
            dcalls_array[dmap[args.d], args.seed] = int(cost)

    print(dcalls_array)





def main():
    #verify_correctness()
    print("FILES VERIFIED\n\n")
    show_plots()


if __name__ == '__main__':
    main()
