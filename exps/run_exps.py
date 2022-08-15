import numpy as np
import math
from typing import Tuple

from utils.constants import (
    BATCH_SIZE,
    ADVERSARIAL_CUSTOM,
    NORMAL_CUSTOM,
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    NORMAL_PAPER,
    UNIFORM_PAPER,
    NETFLIX,
    MOVIE_LENS,
    GREEDY_MIPS,
)
from data.get_data import get_data
from speedup_precision_exps_2 import speedup_precision_exps
from scaling_exp_2 import scaling_exp
from fixed_confidence_exp import fixed_confidence_exp


def delta_epsilon_exp():
    for alg in (ACTION_ELIMINATION, ADAPTIVE_ACTION_ELIMINATION):
        fixed_confidence_exp(
           epsilons=(0.01, 0.8),
           deltas=(0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9),
           num_experiments=20,
           bandit_alg=alg,
           data_type=ADVERSARIAL_CUSTOM,
           with_replacement=False,
        )


def speedup_bandit():

    for alg in (ACTION_ELIMINATION, ADAPTIVE_ACTION_ELIMINATION):
        speedup_precision_exps(
            data_type=NETFLIX,
            mips_alg=alg,
            num_best_atoms=1,
            num_signals=3,
            num_experiments=15,
            num_atoms_list=np.array([10 ** 2, 10 ** 3]),
            len_signals_list=np.array([10 ** 3, 10 ** 4, 10 ** 5])
        )

    for alg in (ACTION_ELIMINATION, ADAPTIVE_ACTION_ELIMINATION):
        speedup_precision_exps(
            data_type=NETFLIX,
            mips_alg=alg,
            num_best_atoms=1,
            num_signals=3,
            num_experiments=15,
            num_atoms_list=np.array([10 ** 2, 10 ** 3]),
            len_signals_list=np.array([10 ** 2, 10 ** 3, 5 * 10 ** 3])
        )


def scaling_experiment_real():
    # Run on Netflix dataset
    for mips_alg in [ADAPTIVE_ACTION_ELIMINATION, ACTION_ELIMINATION, MEDIAN_ELIMINATION, GREEDY_MIPS]:
        scaling_exp(
            0.1,
            0.1,
            num_atoms=10 ** 3,
            size_minmax=(100, 10 ** 5 / 2),
            num_experiments=10,
            num_signals=5,
            data_type=NETFLIX,
            mips_alg=mips_alg,
            with_replacement=False,
            is_normalize=False,
            is_logspace=True,
        )

    # Run on Movie Lens dataset
    for mips_alg in [ADAPTIVE_ACTION_ELIMINATION, ACTION_ELIMINATION, MEDIAN_ELIMINATION, GREEDY_MIPS]:
        scaling_exp(
            0.1,
            0.1,
            num_atoms=10 ** 3,
            size_minmax=(100, 6000),
            num_experiments=10,
            num_signals=5,
            data_type=MOVIE_LENS,
            mips_alg=mips_alg,
            with_replacement=False,
            is_normalize=False,
            is_logspace=True,
        )


if __name__ == "__main__":
    scaling_experiment_real()
    #speedup_bandit()
    #delta_epsilon_exp()
