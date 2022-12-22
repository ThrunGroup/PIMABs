import math
from typing import Tuple
from exps.scaling_exp import scaling_exp
from exps.plot_baselines import generate_scaling_plots, create_scaling_plots
from utils.constants import (
    BATCH_SIZE,
    N,
    D,

    # datasets
    NORMAL_CUSTOM,
    ADVERSARIAL_CUSTOM,
    NETFLIX,
    NETFLIX_TRANSPOSE,
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


def bucket_ae_exp():
    for dataset in [NETFLIX_TRANSPOSE]:
        scaling_exp(
            epsilon=SCALING_EPSILON,
            delta=0.1,
            independent_var=N,
            signal_dim=1000,
            size_minmax=(100, 10 ** 5),
            num_best_atoms=1,
            num_experiments=15,
            num_signals=2,
            data_type=dataset,
            mips_alg=BUCKET_AE,
            with_replacement=False,
            is_normalize=False,
            is_logspace=False,
        )

        scaling_exp(
            epsilon=SCALING_EPSILON,
            delta=0.1,
            independent_var=N,
            signal_dim=1000,
            size_minmax=(100, 10 ** 5),
            num_best_atoms=1,
            num_experiments=15,
            num_signals=2,
            data_type=dataset,
            mips_alg=ACTION_ELIMINATION,
            with_replacement=False,
            is_normalize=False,
            is_logspace=False,
        )
    for dataset in [NETFLIX]:
        scaling_exp(
            epsilon=SCALING_EPSILON,
            delta=0.1,
            independent_var=D,
            num_atoms=1300,
            signal_dim=3000,
            size_minmax=(100, 30000),
            num_best_atoms=1,
            num_experiments=15,
            num_signals=2,
            data_type=dataset,
            mips_alg=BUCKET_AE,
            with_replacement=False,
            is_normalize=False,
            is_logspace=False,
        )

        scaling_exp(
            epsilon=SCALING_EPSILON,
            delta=0.1,
            independent_var=D,
            num_atoms=1300,
            signal_dim=3000,
            size_minmax=(100, 30000),
            num_best_atoms=1,
            num_experiments=15,
            num_signals=2,
            data_type=dataset,
            mips_alg=ACTION_ELIMINATION,
            with_replacement=False,
            is_normalize=False,
            is_logspace=False,
        )
    datasets = [NETFLIX, NETFLIX_TRANSPOSE]
    create_scaling_plots(
        data_types=datasets,
        alg_names=[ACTION_ELIMINATION],
        ind_variables=[N, D],
        include_error_bar=True,
        is_logspace=False,
        is_plot_accuracy=True,
        is_fit=True,
    )
    create_scaling_plots(
        data_types=datasets,
        alg_names=[BUCKET_AE],
        ind_variables=[N, D],
        include_error_bar=True,
        is_logspace=False,
        is_plot_accuracy=True,
        is_fit=True,
    )
    create_scaling_plots(
        data_types=datasets,
        alg_names=[ACTION_ELIMINATION, BUCKET_AE],
        ind_variables=[N, D],
        include_error_bar=True,
        is_logspace=False,
        is_plot_accuracy=True,
        is_fit=False,
    )


if __name__ == "__main__":
    bucket_ae_exp()