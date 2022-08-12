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
    NUM_EXPERIMENTS,
    NUM_POINTS_SCALING
)
from utils.utils import subset_2d, fit_and_plot
from exps.basic_tests import scaling_baseline


def main(
    data_name: str = NORMAL_CUSTOM,
    size_minmax: Tuple[int] = (10, 10 ** 3),
    type: str = ASYMMETRIC
):
    """
    Calls scaling_baseline NUM_EXPERIMENT times and averages the budgets returned.
    Plots the budgets on a linear scale

    :data: either normal_custom, adversarial_custom, or normal_paper
    :type: either symmetric or asymmetric
    """
    print(f"---- running {type}-LSH-MIPS scaling experiments ----")
    print(f"=> using {data_name} dataset")
    print(f"=> averaging over {NUM_EXPERIMENTS} experiments")
    print(f"=> plotting {NUM_POINTS_SCALING} points with signal dimension in range {size_minmax}")

    budgets = np.zeros(NUM_POINTS_SCALING)
    std_array = []
    for ex in range(NUM_EXPERIMENTS):
        print(f"=> running seed {ex + 1}")
        budget, sizes_signal = scaling_baseline(
            seed=ex,
            generate_data=data_name,
            size_minmax=size_minmax,
            num_points=NUM_POINTS_SCALING,
            name_alg=LSH,
            type=type,
        )
        budgets += budget
        std_array.append(budget)

    # sort size_signal based on budget (smallest to largest).
    budgets /= NUM_EXPERIMENTS
    functions = [np.array]
    function_names = ["Linear"]
    std = np.std(std_array) / np.sqrt(NUM_EXPERIMENTS)
    fit_and_plot(
        xs=sizes_signal, ys=budgets, ys_std=std
    )


if __name__ == "__main__":
    # run asymmetric lsh
    main(data_name=NORMAL_CUSTOM, size_minmax=(10, 10 ** 4), type=ASYMMETRIC)
    main(data_name=ADVERSARIAL_CUSTOM, size_minmax=(10, 10 ** 4), type=ASYMMETRIC)
    main(data_name=NORMAL_PAPER, size_minmax=(10, 10 ** 4), type=ASYMMETRIC)

    # run symmetric lsh
    main(data_name=NORMAL_CUSTOM, size_minmax=(10, 10 ** 4), type=SYMMETRIC)
    main(data_name=ADVERSARIAL_CUSTOM, size_minmax=(10, 10 ** 4), type=SYMMETRIC)
    main(data_name=NORMAL_PAPER, size_minmax=(10, 10 ** 4), type=SYMMETRIC)