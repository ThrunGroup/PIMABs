import numpy as np
import math
import pandas as pd
import os
from typing import Tuple, List

from utils.constants import (
    ADVERSARIAL_CUSTOM,
    NORMAL_CUSTOM,
    NORMAL_PAPER,
    NUM_EXPERIMENTS,
    NUM_POINTS_PCA,
    BUFFER,
    PCA,
)
from utils.utils import subset_2d, fit_and_plot
from exps.basic_tests import precision_speed_pca



def main(
    data_name: str = NORMAL_CUSTOM,
    top_k: List[int] = [5, 10],
    depth_minmax: Tuple[int] = (0, 5),
):
    """
    Plots the precision, speed tradeoff of pca-mips. Precision is defined as the fraction of true top K solutions
    in the returned top K solutions and speed is dictated by the total budget used. Results are plotted.

    :data_name: Either normal_custom, adversarial_custom, or normal_paper
    :depth_minmax: Minmax of depth of pca-tree
    """
    print("---- running PCA-MIPS precision experiments ----")
    print(f"=> using {data_name} dataset")
    print(f"=> averaging over {NUM_EXPERIMENTS} experiments")
    print(f"=> plotting {NUM_POINTS_PCA} points with depth in range {depth_minmax}")
    print(f"=> finding top {top_k} candidates")

    budgets = np.zeros((NUM_POINTS_PCA, len(top_k)))
    precisions = np.zeros((NUM_POINTS_PCA, len(top_k)))

    for ex in range(NUM_EXPERIMENTS):
        print(f"... running seed {ex + 1}")
        budget, precision = precision_speed_pca(
            seed=ex,
            top_k=top_k,
            generate_data=data_name,
            num_points=NUM_POINTS_PCA,
            depth_minmax=depth_minmax,
        )
        budgets += budget
        precisions += precision

    budgets += BUFFER
    precisions += BUFFER
    budgets /= NUM_EXPERIMENTS
    precisions /= NUM_EXPERIMENTS

    # sort precision based on "largest" budget. Note that speed is the inverse of budget
    for i in range(len(top_k)):
        k_budget = budgets[:, i]
        k_precision = precisions[:, i]
        sorted_by_budget = np.argsort(-k_budget)
        log_dict = {
            "data_type": data_name,
            "budgets": k_budget[sorted_by_budget],
            "accuracy": k_precision[sorted_by_budget],
        }
        log_df = pd.DataFrame(log_dict)
        log_dir = "logs"
        filename = os.path.join(
            log_dir,
            PCA
            + "_"
            + "SPEEDUP_"
            + data_name
            + "_"
            + str(depth_minmax)
            + ".csv"
        )
        os.makedirs(log_dir, exist_ok=True)
        log_df.to_csv(filename, index=False)


if __name__ == "__main__":
    main(data_name=NORMAL_CUSTOM, top_k=[10], depth_minmax=(0, 4))
    main(data_name=ADVERSARIAL_CUSTOM, top_k=[10], depth_minmax=(0, 4))
    main(data_name=NORMAL_PAPER, top_k=[10], depth_minmax=(0, 4))
