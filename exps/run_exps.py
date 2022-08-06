import numpy as np
import math
from typing import Tuple

from mips.utils.constants import (
    BATCH_SIZE,
    ADVERSARIAL_CUSTOM,
    NORMAL_CUSTOM,
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    HOEFFDING,
)
from utils.utils import subset_2d, fit_and_plot
from utils.bandit_utils import get_ci
from data.custom_data import generate_normal_data, generate_adversarial_data


if __name__ == "__main__":
    epsilons = (0.05, 0.1, 0.2, 0.3)
    deltas = (
        0.05,
        0.1,
        0.2,
        0.3,
    )
    # fixed_confidence_exp(
    #     epsilons=epsilons,
    #     deltas=deltas,
    #     bandit_alg=ACTION_ELIMINATION,
    #     num_experiments=50,
    #     with_replacement=True,
    # )
    scaling_exp(
        0.05,
        0.1,
        size_minmax=(10, 10 ** 5),
        num_experiments=10,
        generate_data=NORMAL_CUSTOM,
        bandit_alg=ACTION_ELIMINATION,
        with_replacement=False,
        is_normalize=False,
    )
    scaling_exp(
        0.05,
        0.1,
        size_minmax=(10, 10 ** 5),
        num_experiments=10,
        generate_data=NORMAL_CUSTOM,
        bandit_alg=ACTION_ELIMINATION,
        with_replacement=True,
        is_normalize=False,
    )
    scaling_exp(
        0.05,
        0.1,
        size_minmax=(10, 10 ** 5),
        num_experiments=10,
        generate_data=NORMAL_CUSTOM,
        bandit_alg=MEDIAN_ELIMINATION,
        is_normalize=False,
    )
    # fixed_confidence_exp(
    #     epsilons=epsilons,
    #     deltas=deltas,
    #     bandit_alg=ACTION_ELIMINATION,
    #     num_experiments=10,
    #     with_replacement=False,
    # )
