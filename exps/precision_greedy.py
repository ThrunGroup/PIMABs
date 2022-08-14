import numpy as np
import math
import pandas as pd
import os
from typing import Tuple, List

from utils.constants import (
    ADVERSARIAL_CUSTOM,
    NORMAL_CUSTOM,
    NORMAL_PAPER,
    ASYMMETRIC,
    SYMMETRIC,
    NUM_EXPERIMENTS,
    NUM_POINTS_LSH,
    NUM_HFUNC,
    BUFFER,
)
from utils.utils import subset_2d, fit_and_plot
from exps.basic_tests import precision_speed_lsh


def main(
    data_name: str = NORMAL_CUSTOM,
    top_k: List[int] = [5, 10],
    hfunc_minmax: Tuple[int] = (0, NUM_HFUNC),
    table_minmax: Tuple[int] = (1, 4),
    type: str = ASYMMETRIC,
):
    """
    Plots the precision, speed tradeoff of lsh-mips. Precision is defined as the fraction of true top K solutions
    in the returned top K solutions and speed is dictated by the total budget used. Results are plotted.

    :data_name: Either normal_custom, adversarial_custom, or normal_paper
    :hfunc_minmax: Min max of number of hyperplanes
    :table_minmax: Minmax of table size.
    :type: Either asymmetric (2014) or symmetric (2015)
    """
    print(f"---- running {type}-LSH-MIPS precision experiments ----")
    print(f"=> using {data_name} dataset")
    print(f"=> averaging over {NUM_EXPERIMENTS} experiments")
    print(f"=> plotting {NUM_POINTS_LSH} points with tables in range {table_minmax}")
    print(f"=> finding top {top_k} candidates")

    budgets = np.zeros((NUM_POINTS_LSH, len(top_k)))
    precisions = np.zeros((NUM_POINTS_LSH, len(top_k)))

    for ex in range(NUM_EXPERIMENTS):
        print(f"\n=> running seed {ex + 1}")
        budget, precision = precision_speed_lsh(
            seed=ex,
            top_k=top_k,
            generate_data=data_name,
            num_points=NUM_POINTS_LSH,
            hfunc_minmax=hfunc_minmax,
            table_minmax=table_minmax,
            type=type,
        )

        # shape (NUM_POINTS_LSH, len(top_k))
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
            type
            + "_LSH_MIPS"
            "_SPEEDUP_"
            + data_name
            + "_"
            + str(hfunc_minmax)
            + "_"
            + str(table_minmax)
            + ".csv"
        )
        os.makedirs(log_dir, exist_ok=True)
        log_df.to_csv(filename, index=False)


if __name__ == "__main__":
    # run asymmetric lsh
    main(data_name=NORMAL_CUSTOM, top_k=[10], hfunc_minmax=(0, NUM_HFUNC), table_minmax=(1, 3), type=ASYMMETRIC)
    main(
         ADVERSARIAL_CUSTOM,
         top_k=[5],
         hfunc_minmax=(0, NUM_HFUNC + 5),
         table_minmax=(1, 2),
         type=ASYMMETRIC,
    )
    main(NORMAL_PAPER, top_k=[10], hfunc_minmax=(0, NUM_HFUNC), table_minmax=(1, 2), type=ASYMMETRIC)

    # run symmetric lsh
    main(NORMAL_CUSTOM, top_k=[10], hfunc_minmax=(0, NUM_HFUNC-5), table_minmax=(1, 4), type=SYMMETRIC)
    main(ADVERSARIAL_CUSTOM, top_k=[10], hfunc_minmax=(0, NUM_HFUNC-1), table_minmax=(1, 3), type=SYMMETRIC)
    main(NORMAL_PAPER, top_k=[10], hfunc_minmax=(0, NUM_HFUNC-4), table_minmax=(1, 4), type=SYMMETRIC)
