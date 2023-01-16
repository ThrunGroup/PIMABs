import numpy as np
import math
import os
from typing import Tuple

from exps.scaling_exp import scaling_exp
from exps.plot_baselines import create_scaling_plots
from utils.constants import (
    BATCH_SIZE,

    #datasets
    SIMPLE_SONG,

    # algorithms
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    GREEDY_MIPS,
    LSH_MIPS,
    PCA_MIPS,

    # scaling specific constants
    SCALING_NUM_EXPERIMENTS,
    SCALING_NUM_ATOMS,
    SCALING_NUM_SIGNALS,
    SCALING_EPSILON,
    SCALING_DELTA,
    SCALING_SIZE_MINMAX,
    SCAlING_SIZE_MINMAX_MOVIE,
    SCALING_DELTA_REAL,
    SCALING_EPSILON_REAL,
)


def run_high_dim_scaling(run: bool = True, plot: bool = True):
    """
    Run scaling experiments for the toy datasets
    """
    if run:
        for data_type in [SIMPLE_SONG]:
            mean_acc = scaling_exp(
                epsilon=0.00,
                delta=0.0001,
                num_atoms=10,
                size_minmax=(1e6, 26e6),
                num_experiments=3,
                num_signals=1,
                data_type=SIMPLE_SONG,
                mips_alg=ACTION_ELIMINATION,
                with_replacement=True,
                is_normalize=False,
                is_logspace=True,
                dirname="high_dim_song_scaling_logs",
                num_seeds=3,
            )
            assert mean_acc >= 0.97, f"Mean accuracy on dataset {data_type} should be above 97\%"

    if plot:
        create_scaling_plots(
            data_types=[SIMPLE_SONG],
            alg_names=[ACTION_ELIMINATION],
            include_error_bar=True,
            is_fit=True,
            is_logspace=True,
            dirname="high_dim_song_scaling_logs",
        )

if __name__ == "__main__":
    run_high_dim_scaling(run=True, plot=True)