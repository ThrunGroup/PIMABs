import numpy as np
import math
from typing import Tuple
from exps.scaling_exp import scaling_exp
from exps.plot_baselines import generate_scaling_plots, create_scaling_plots
from utils.constants import (
    BATCH_SIZE,

    # datasets
    NORMAL_CUSTOM,
    ADVERSARIAL_CUSTOM,
    NORMAL_PAPER,
    NETFLIX,
    MOVIE_LENS,
    COR_NORMAL_CUSTOM,

    # algorithms
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    GREEDY_MIPS,
    LSH_MIPS,
    PCA_MIPS,
    BUCKET_AE,

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


def run_scaling():
    """
    Run scaling experiment for the two synthetic datasets (normal custom, adversarial custom) and
    the two real-world datsets (netflix and movie lens). This function is called by repro_script_python.py.
    """
    # Get data for the scaling experiments for the datasets (three synthetic datasets)
    for data_type in [NORMAL_CUSTOM, ADVERSARIAL_CUSTOM, NETFLIX, MOVIE_LENS]:
        if data_type is MOVIE_LENS:
            size_minmax = SCAlING_SIZE_MINMAX_MOVIE
        else:
            size_minmax = SCALING_SIZE_MINMAX

        if data_type in [NORMAL_CUSTOM, ADVERSARIAL_CUSTOM]:
            epsilon = SCALING_EPSILON
            delta = SCALING_DELTA
        else:  # real datasets
            epsilon = SCALING_EPSILON_REAL
            delta = SCALING_DELTA_REAL

        scaling_exp(
            epsilon=epsilon,
            delta=delta,
            num_atoms=SCALING_NUM_ATOMS,
            size_minmax=size_minmax,
            num_experiments=SCALING_NUM_EXPERIMENTS,
            num_signals=SCALING_NUM_SIGNALS,
            data_type=data_type,
            mips_alg=ADAPTIVE_ACTION_ELIMINATION,
            with_replacement=False,
            is_normalize=False,
            is_logspace=True,
        )
        scaling_exp(
            epsilon=epsilon,
            delta=delta,
            num_atoms=SCALING_NUM_ATOMS,
            size_minmax=size_minmax,
            num_experiments=SCALING_NUM_EXPERIMENTS,
            num_signals=SCALING_NUM_SIGNALS,
            data_type=data_type,
            mips_alg=ACTION_ELIMINATION,
            with_replacement=False,
            is_normalize=False,
            is_logspace=True,
        )
        scaling_exp(
            epsilon=epsilon,
            delta=delta,
            num_atoms=SCALING_NUM_ATOMS,
            size_minmax=size_minmax,
            num_experiments=SCALING_NUM_EXPERIMENTS,
            num_signals=SCALING_NUM_SIGNALS,
            data_type=data_type,
            mips_alg=MEDIAN_ELIMINATION,
            with_replacement=False,
            is_normalize=False,
            is_logspace=True,
        )
        scaling_exp(
            epsilon=epsilon,
            delta=delta,
            num_atoms=SCALING_NUM_ATOMS,
            size_minmax=size_minmax,
            num_experiments=SCALING_NUM_EXPERIMENTS,
            num_signals=SCALING_NUM_SIGNALS,
            data_type=data_type,
            mips_alg=GREEDY_MIPS,
            with_replacement=False,
            is_normalize=False,
            is_logspace=True,
        )
        scaling_exp(
            epsilon=epsilon,
            delta=delta,
            num_atoms=SCALING_NUM_ATOMS,
            size_minmax=size_minmax,
            num_experiments=SCALING_NUM_EXPERIMENTS,
            num_signals=SCALING_NUM_SIGNALS,
            data_type=data_type,
            mips_alg=PCA_MIPS,
            with_replacement=False,
            is_normalize=False,
            is_logspace=True,
        )
        scaling_exp(
            epsilon=epsilon,
            delta=delta,
            num_atoms=SCALING_NUM_ATOMS,
            size_minmax=size_minmax,
            num_experiments=SCALING_NUM_EXPERIMENTS,
            num_signals=SCALING_NUM_SIGNALS,
            data_type=data_type,
            mips_alg=LSH_MIPS,
            with_replacement=False,
            is_normalize=False,
            is_logspace=True,
        )


if __name__ == "__main__":
    datasets = [NETFLIX, NORMAL_CUSTOM, ADVERSARIAL_CUSTOM, COR_NORMAL_CUSTOM]
    for dataset in [NORMAL_CUSTOM]:
        scaling_exp(
            epsilon=SCALING_EPSILON,
            delta=SCALING_DELTA,
            num_atoms=1300,
            size_minmax=(100, 30000),
            num_experiments=SCALING_NUM_EXPERIMENTS,
            num_signals=SCALING_NUM_SIGNALS,
            data_type=dataset,
            mips_alg=BUCKET_AE,
            with_replacement=False,
            is_normalize=False,
            is_logspace=True,
        )

        scaling_exp(
            epsilon=SCALING_EPSILON,
            delta=SCALING_DELTA,
            num_atoms=1300,
            size_minmax=(10, 30000),
            num_experiments=SCALING_NUM_EXPERIMENTS,
            num_signals=SCALING_NUM_SIGNALS,
            data_type=dataset,
            mips_alg=ACTION_ELIMINATION,
            with_replacement=False,
            is_normalize=False,
            is_logspace=True,
        )
    create_scaling_plots(
        data_types=datasets,
        alg_names =[ACTION_ELIMINATION, BUCKET_AE],
        include_error_bar=True,
        is_logspace=True,
        is_plot_accuracy=True,
        is_fit=False,
    )
