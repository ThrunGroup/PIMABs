import numpy as np
import math
from typing import Tuple

from utils.constants import (
    ADVERSARIAL_CUSTOM,
    NORMAL_CUSTOM,
    NORMAL_PAPER,
    PCA,
    NUM_EXPERIMENTS
)
from utils.utils import subset_2d, fit_and_plot
from exps.basic_tests import scaling_baseline

def main(data:str = NORMAL_CUSTOM):
    budgets = np.zeros((NUM_EXPERIMENTS, 1))
    for ex in range(NUM_EXPERIMENTS):
        print(f"=> running ex {ex + 1}")
        budget, sizes_signal = scaling_baseline(
            num_experiments=NUM_EXPERIMENTS,
            generate_data=data,
            name_alg=PCA,
            is_normalize=False,
        )
        budgets += budget

    functions = [np.array]
    function_names = ["Linear"]
    fit_and_plot(sizes_signal, budgets/NUM_EXPERIMENTS, functions, function_names)


if __name__ == "__main__":
    main(NORMAL_CUSTOM)
    main(ADVERSARIAL_CUSTOM)
    main(NORMAL_PAPER)
