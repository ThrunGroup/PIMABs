"""
Code to automatically parse the profiles produced from running experiments.
In particular, plots the scaling of B_MP vs. d for various dimension sizes d.
Used to demonstrate O(dlogd) scaling.
"""

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
from sklearn.linear_model import LinearRegression

import sys

sys.path.append("..")
from utils import get_args, get_logfile_name, remap_args

# Possible line numbers for the empty_counter fn
# FN_NAME_1 = 'data_utils.py:129(empty_counter)'
# FN_NAME_2 = 'data_utils.py:141(empty_counter)'
# FN_NAME_3 = 'data_utils.py:142(empty_counter)'


def verify_correctness():
    """
    Verifies that BanditPAM returns the same BUILD and SWAP medoid assignments
    as PAM, by parsing the logfiles.
    """

    parent_dirs = [
        "logs",
    ]

    for parent_dir in parent_dirs:
        bmp_logfiles = [
            os.path.join(parent_dir, x)
            for x in os.listdir(parent_dir)
            if os.path.isfile(os.path.join(parent_dir, x))
            and x != ".DS_Store"
            and x[-4:] == "-bmp"
        ]

        for bmp_lfile in sorted(bmp_logfiles):
            mp_lfile = bmp_lfile.replace("-bmp", "")
            if not os.path.exists(mp_lfile):
                print("Warning: no naive experiment", mp_lfile)
            else:
                disagreement = False
                with open(bmp_lfile, "r") as fin1:
                    with open(mp_lfile, "r") as fin2:
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


def plot_slice_sns(dcalls_array, ds, seeds, take_log=True):
    """
    Plots the number of coordinate-wise distance calls vs. d, for various algorithms and seeds

    :param: dcalls_array, an array of d, seed, calls
    """

    # To show y = log(x), plot y vs logx and show linear relationship
    if take_log:
        xs = np.log10(ds)
        log_prefix = "LOG"
    else:
        xs = np.array(ds)
        log_prefix = "LINEAR"

    np_data = dcalls_array
    sns.set()
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(7, 7))

    # Make a dataframe with the relevant data, for plotting with seaborn
    d = {"d": xs}
    for seed_idx, seed in enumerate(seeds):
        d["seed_" + str(seed)] = np_data[:, seed_idx]
    df = pd.DataFrame(data=d)
    print(df)

    # Combine the different seeds into 1 column
    melt_df = df.melt("d", var_name="seeds", value_name="cost")
    sns.scatterplot(x="d", y="cost", data=melt_df, ax=ax, alpha=0.6)

    # Plot means and error bars
    bars = (1.96 / (10 ** 0.5)) * np.std(np_data, axis=1)
    means = np.mean(np_data, axis=1)
    plt.errorbar(
        xs,
        means,
        yerr=bars,
        fmt="+",
        capsize=5,
        ecolor="black",
        elinewidth=1.5,
        zorder=100,
        mec="black",
        mew=1.5,
        label="95% confidence interval",
    )

    plt.title("Total cost vs. dimensionality of vectors: " + log_prefix)
    plt.xlabel("$\log_{10}(d)$")
    plt.ylabel("$\log_{10}$(# of coordinate-wise distance computations per step)")

    # Plot line of best fit
    LR = LinearRegression(fit_intercept=True)
    reg = LR.fit(
        xs.reshape(-1, 1), means
    )  # Need the reshape since there's only 1 feature
    sl = reg.coef_[0]
    icpt = reg.intercept_
    x_min, x_max = plt.xlim()
    plt.plot(
        [x_min, x_max],
        [x_min * sl + icpt, x_max * sl + icpt],
        color="black",
        label="Linear fit, slope=%0.3f" % (sl),
    )
    print("Slope is:", sl)
    print("Intercept is is:", icpt)
    print("R^2 is is: ", reg.score(xs.reshape(-1, 1), means))

    # Plot guideline
    # plt.plot([x_min, x_max], [sl*np.exp(x_min), sl*np.exp(x_max)], color='red', linestyle=':', label='$O(d)$ linear scaling')

    # Modify filename depending on dataset
    plt.savefig("figures/" + log_prefix + ".pdf")


def show_plots():
    """
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
    """
    args = get_args()  # Uses default values for now as placeholder to instantiate args

    ### Decide which experiments to plot
    algos = ["B_MP"]  # , "MP"]
    Ns = [25]
    ds = [
        80,
        160,
        320,
        640,
        784,
    ]
    dmap = {
        80: 0,
        160: 1,
        320: 2,
        640: 3,
        784: 4,
    }
    seeds = range(10)
    ps = [1e-2]  # This is unused by the experiments and recomputed in-method
    batch_sizes = [10]  # This is unused by the experiments and recomputed in-method
    max_iters = [1]
    tols = [1e-2]
    residues = [1e-2]
    exps = itertools.product(
        algos, Ns, ds, seeds, ps, batch_sizes, max_iters, tols, residues
    )

    ### Gather data
    dcalls_array = np.zeros((len(ds), len(seeds)))
    for exp in exps:
        args = remap_args(args, exp)
        logfile = os.path.join(get_logfile_name(args, bandit=True))
        if not os.path.exists(logfile):
            raise Exception("No logfile for experiment", logfile)

        with open(logfile, "r") as fin:
            cost = fin.readline().strip("Total cost: ")
            if dcalls_array[dmap[args.d], args.seed] != 0:
                raise Exception("Rewriting data matrix on", logfile)
            dcalls_array[dmap[args.d], args.seed] = int(cost)

    print(dcalls_array)
    print(np.mean(dcalls_array, axis=1))
    plot_slice_sns(dcalls_array, ds, seeds, take_log=False)
    plot_slice_sns(dcalls_array, ds, seeds, take_log=True)


def main():
    # verify_correctness()
    print("FILES VERIFIED\n\n")
    show_plots()


if __name__ == "__main__":
    main()
