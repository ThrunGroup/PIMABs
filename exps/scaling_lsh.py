import numpy as np
import math
from typing import Tuple

from utils.constants import (
    ADVERSARIAL_CUSTOM,
    NORMAL_CUSTOM,
    NORMAL_PAPER,
    LSH,
    ASYMMETRIC,
    SYMMETRIC,
    NUM_EXPERIMENTS
)
from utils.utils import subset_2d, fit_and_plot
from exps.basic_tests import scaling_baseline


def main(data:str = NORMAL_CUSTOM, type:str = ASYMMETRIC):
    """
    Calls scaling_baseline NUM_EXPERIMENT times and averages the budgets returned.
    Plots the budgets on a linear scale

    :data: either normal_custom, adversarial_custom, or normal_paper
    :type: either symmetric or asymmetric
    """
    budgets = np.zeros((NUM_EXPERIMENTS, 1))
    for ex in range(NUM_EXPERIMENTS):
        print(f"=> running ex {ex + 1}")
        budget, sizes_signal = scaling_baseline(
            num_experiments=NUM_EXPERIMENTS,
            generate_data=data,
            name_alg=LSH,
            type=type,
            is_normalize=False,
        )
        budgets += budget

    functions = [np.array]
    function_names = ["Linear"]
    fit_and_plot(sizes_signal, budgets/NUM_EXPERIMENTS, functions, function_names)


if __name__ == "__main__":
    # run asymmetric lsh
    main(NORMAL_CUSTOM, type=ASYMMETRIC)
    main(ADVERSARIAL_CUSTOM, type=ASYMMETRIC)
    main(NORMAL_PAPER, type=ASYMMETRIC)

    # run symmetric lsh
    main(NORMAL_CUSTOM, type=SYMMETRIC)
    main(ADVERSARIAL_CUSTOM, type=SYMMETRIC)
    main(NORMAL_PAPER, type=SYMMETRIC)