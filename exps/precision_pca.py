import numpy as np
import math
from typing import Tuple

from utils.constants import (
    ADVERSARIAL_CUSTOM,
    NORMAL_CUSTOM,
    NORMAL_PAPER,
    PCA,
    NUM_EXPERIMENTS,
    BUFFER
)
from utils.utils import subset_2d, fit_and_plot
from exps.basic_tests import precision_speed


def main(data:str = NORMAL_CUSTOM):
    budgets = np.zeros((NUM_EXPERIMENTS, 1))
    precisions = np.zeros((NUM_EXPERIMENTS, 1))
    for ex in range(NUM_EXPERIMENTS):
        print(f"=> running ex {ex + 1}")
        budget, precision = precision_speed(
            num_experiments=NUM_EXPERIMENTS,
            generate_data=data,
            name_alg=PCA,
            is_normalize=False,
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
    main(NORMAL_CUSTOM)
    main(ADVERSARIAL_CUSTOM)
    main(NORMAL_PAPER)
