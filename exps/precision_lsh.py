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
    BUFFER
)
from utils.utils import subset_2d, fit_and_plot
from exps.basic_tests import precision_speed


def main(data:str = NORMAL_CUSTOM, type:str = ASYMMETRIC):
    budgets = np.zeros((NUM_EXPERIMENTS, 1))
    precisions = np.zeros((NUM_EXPERIMENTS, 1))
    for ex in range(NUM_EXPERIMENTS):
        print(f"=> running ex {ex + 1}")
        budget, precision = precision_speed(
            num_experiments=NUM_EXPERIMENTS,
            generate_data=data,
            name_alg=LSH,
            is_normalize=False,
            type=type,
            num_hfunc=10,
            num_tables=1,
        )
        budgets += budget
        precisions += precision

    functions = [np.log]
    function_names = ["Logarithm"]

    budgets += BUFFER
    precisions += BUFFER
    budgets /= NUM_EXPERIMENTS
    precisions /= NUM_EXPERIMENTS
    fit_and_plot(
        precisions, np.flip(budgets), functions, function_names
    )


if __name__ == "__main__":
    # run asymmetric lsh
    main(NORMAL_CUSTOM, type=ASYMMETRIC)
    main(ADVERSARIAL_CUSTOM, type=ASYMMETRIC)
    main(NORMAL_PAPER, type=ASYMMETRIC)

    # run symmetric lsh
    main(NORMAL_CUSTOM, type=SYMMETRIC)
    main(ADVERSARIAL_CUSTOM, type=SYMMETRIC)
    main(NORMAL_PAPER, type=SYMMETRIC)
