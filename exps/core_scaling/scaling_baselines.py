import numpy as np
import math
from typing import Tuple
from exps.scaling_exp import scaling_exp
from exps.plot_baselines import generate_scaling_plots, create_scaling_plots
from utils.constants import (
    BATCH_SIZE,

    # datasets
    NORMAL_CUSTOM,
    UNIFORM_PAPER,
    NETFLIX,
    MOVIE_LENS,
    CRYPTO_PAIRS,
    COR_NORMAL_CUSTOM,
    CLEAR_LEADER_HARD,
    NO_CLEAR_LEADER,

    # algorithms
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    GREEDY_MIPS,
    LSH_MIPS,
    PCA_MIPS,
    HNSW_MIPS,
    NAPG_MIPS,
    NEQ_MIPS,
    BUCKET_ACTION_ELIMINATION,
    H2ALSH,

    # scaling specific constants
    SCALING_NUM_EXPERIMENTS,
    SCALING_NUM_SIGNALS,
    SCALING_NUM_ATOMS,
    SCALING_NUM_ATOMS_CRYPTO_PAIRS,
    SCALING_EPSILON,
    SCALING_DELTA,
    SCALING_SIZE_MINMAX,
    SCAlING_SIZE_MINMAX_MOVIE,
    SCALING_DELTA_REAL,
    SCALING_EPSILON_REAL,
    DIMENSION_OF_ATOMS,
    LARGE_DELTA,
)


def get_scaling_baseline_params(data_type):
    """
    Function gives the constants for [epsilon, delta, size_minmax] for the specified datatype.
    """
    if data_type is UNIFORM_PAPER:
        epsilon = SCALING_EPSILON
        delta = SCALING_DELTA
    else:  # real datasets
        epsilon = SCALING_EPSILON_REAL
        delta = SCALING_DELTA_REAL

    # Crypto-pairs requires a very small delta since variance is big
    if data_type is CRYPTO_PAIRS:
        delta = LARGE_DELTA

    # movie_lens dataset has a smaller size than the rest
    if data_type is MOVIE_LENS:
        size_minmax = SCAlING_SIZE_MINMAX_MOVIE
    else:
        size_minmax = SCALING_SIZE_MINMAX

    return epsilon, delta, size_minmax


def scaling_baselines():
    """
    Run scaling experiments for the 5 datasets (2 synthetic, 3 real) on 9 baseline MIPS algorithms.
    This function is called by repro_script_python.py.
    """
    # Get data for the scaling experiments for the datasets (not the high-dimensional datasets)
    for data_type in [NORMAL_CUSTOM, UNIFORM_PAPER, CLEAR_LEADER_HARD, NETFLIX, MOVIE_LENS]:
        # for algorithm in [
        #     GREEDY_MIPS,  # greedy approach
        #     LSH_MIPS, H2ALSH,  # mapping to higher dimensions
        #     PCA_MIPS, NEQ_MIPS, NAPG_MIPS,  # mapping to lower dimensions
        #     MEDIAN_ELIMINATION, ACTION_ELIMINATION, ADAPTIVE_ACTION_ELIMINATION,  # bandit approach
        # ]:
        for algorithm in [NAPG_MIPS]:
            epsilon, delta, size_minmax = get_scaling_baseline_params(data_type)
            scaling_exp(
                epsilon=epsilon,
                delta=delta,
                num_atoms=SCALING_NUM_ATOMS,
                size_minmax=size_minmax,
                num_experiments=SCALING_NUM_EXPERIMENTS,
                num_signals=SCALING_NUM_SIGNALS,
                data_type=data_type,
                mips_alg=algorithm,
                with_replacement=False,
                is_normalize=False,
                is_logspace=True,
            )


if __name__ == "__main__":
    scaling_baselines()
