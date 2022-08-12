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
from scaling_exp import scaling_exp
from fixed_confidence_exp import fixed_confidence_exp


if __name__ == "__main__":
    epsilons = (0.05, 0.1, 0.2, 0.3)
    deltas = (
        0.05,
        0.1,
        0.2,
        0.3,
    )
    fixed_confidence_exp(
        epsilons=epsilons,
        deltas=deltas,
        bandit_alg=ACTION_ELIMINATION,
        num_experiments=50,
        with_replacement=True,
    )
    scaling_exp(
        0,
        0.5,
        size_minmax=(10, 10 ** 4),
        num_experiments=5,
        num_signals=5,
        data_type=NORMAL_CUSTOM,
        mips_alg=ADAPTIVE_ACTION_ELIMINATION,
        with_replacement=False,
        is_normalize=False,
    )
    scaling_exp(
        0,
        0.5,
        size_minmax=(10, 10 ** 5),
        num_experiments=10,
        data_type=UNIFORM_PAPER,
        algorithm=ACTION_ELIMINATION,
        with_replacement=False,
        is_normalize=False,
    )
    scaling_exp(
        0,
        0.5,
        size_minmax=(50, 10 ** 4),
        num_experiments=5,
        num_signals=5,
        data_type=NORMAL_CUSTOM,
        mips_alg=ACTION_ELIMINATION,
        with_replacement=True,
        is_normalize=False,
    )
    scaling_exp(
        0.05,
        0.1,
        size_minmax=(30, 10 ** 4),
        num_signals=5,
        num_experiments=5,
        data_type=NORMAL_CUSTOM,
        mips_alg=MEDIAN_ELIMINATION,
        is_normalize=False,
    )

    # fixed_confidence_exp(
    #     epsilons=epsilons,
    #     deltas=deltas,
    #     bandit_alg=ACTION_ELIMINATION,
    #     num_experiments=10,
    #     with_replacement=False,
    # )
