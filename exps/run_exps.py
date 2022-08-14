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
)
from speedup_precision_exps import speedup_precision_exps
from scaling_exp import scaling_exp
from fixed_confidence_exp import fixed_confidence_exp


if __name__ == "__main__":
    # epsilons = (0.05, 0.1, 0.2, 0.3)
    # deltas = (
    #     0.05,
    #     0.1,
    #     0.2,
    #     0.3,
    # )
    # fixed_confidence_exp(
    #     epsilons=epsilons,
    #     deltas=deltas,
    #     bandit_alg=ACTION_ELIMINATION,
    #     num_experiments=50,
    #     with_replacement=True,
    # )
    # speedup_precision_exps(
    #     data_type=NORMAL_CUSTOM,
    #     mips_alg=ADAPTIVE_ACTION_ELIMINATION,
    #     num_signals=3,
    # )
    # scaling_exp(
    #     0,
    #     0.5,
    #     num_atoms=10 ** 4,
    #     size_minmax=(10, 10 ** 4),
    #     num_experiments=10,
    #     num_signals=3,
    #     data_type=NORMAL_PAPER,
    #     mips_alg=ADAPTIVE_ACTION_ELIMINATION,
    #     with_replacement=False,
    #     is_normalize=False,
    # )
    # scaling_exp(
    #     0,
    #     0.5,
    #     num_atoms=10 ** 4,
    #     size_minmax=(10, 10 ** 4),
    #     num_experiments=10,
    #     num_signals=3,
    #     data_type=NORMAL_PAPER,
    #     mips_alg=ACTION_ELIMINATION,
    #     with_replacement=False,
    #     is_normalize=False,
    # )
    # scaling_exp(
    #     0,
    #     0.5,
    #     size_minmax=(50, 10 ** 4),
    #     num_experiments=5,
    #     num_signals=5,
    #     data_type=NORMAL_CUSTOM,
    #     mips_alg=ACTION_ELIMINATION,
    #     with_replacement=True,
    #     is_normalize=False,
    # )
    scaling_exp(
        0.05,
        0.1,
        num_atoms=int(10**2 / 3),
        size_minmax=(100, 10**6),
        num_signals=3,
        num_experiments=100,
        data_type=NORMAL_CUSTOM,
        mips_alg=MEDIAN_ELIMINATION,
        is_normalize=False,
        is_logspace=False,
    )

    # fixed_confidence_exp(
    #     epsilons=epsilons,
    #     deltas=deltas,
    #     bandit_alg=ACTION_ELIMINATION,
    #     num_experiments=10,
    #     with_replacement=False,
    # )
