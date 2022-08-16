import numpy as np
import math
from typing import Tuple
from speedup_precision_exps import speedup_precision_exps
from utils.constants import (
    BATCH_SIZE,

    # datasets
    NORMAL_CUSTOM,
    ADVERSARIAL_CUSTOM,
    NORMAL_PAPER,

    # algorithms
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    GREEDY_MIPS,
    LSH_MIPS,
    PCA_MIPS,

    # scaling specific constants
    TRADEOFF_NUM_EXPERIMENTS,
    TRADEOFF_NUM_ATOMS,
    TRADEOFF_NUM_SIGNALS,
    TRADEOFF_PCA_NUM_EXPERIMENTS,
    TRADEOFF_LSH_NUM_EXPERIMENTS,
)


if __name__ == "__main__":
    # Get data for the scaling experiments for the datasets (three synthetic datasets)
    for data_type in [NORMAL_PAPER]:
        speedup_precision_exps(
            num_atoms=TRADEOFF_NUM_ATOMS,
            num_experiments=TRADEOFF_NUM_EXPERIMENTS,
            num_signals=TRADEOFF_NUM_SIGNALS,
            data_type=data_type,
            mips_alg=ADAPTIVE_ACTION_ELIMINATION,
            with_replacement=False,
        )
        # speedup_precision_exps(
        #     num_atoms=TRADEOFF_NUM_ATOMS,
        #     num_experiments=TRADEOFF_NUM_EXPERIMENTS,
        #     num_signals=TRADEOFF_NUM_SIGNALS,
        #     data_type=data_type,
        #     mips_alg=ACTION_ELIMINATION,
        #     with_replacement=False,
        # )
        # speedup_precision_exps(
        #     num_atoms=TRADEOFF_NUM_ATOMS,
        #     num_experiments=TRADEOFF_NUM_EXPERIMENTS,
        #     num_signals=TRADEOFF_NUM_SIGNALS,
        #     data_type=data_type,
        #     mips_alg=MEDIAN_ELIMINATION,
        #     with_replacement=False,
        # )
        # speedup_precision_exps(
        #     num_atoms=TRADEOFF_NUM_ATOMS,
        #     num_experiments=TRADEOFF_NUM_EXPERIMENTS,
        #     num_signals=TRADEOFF_NUM_SIGNALS,
        #     data_type=data_type,
        #     mips_alg=GREEDY_MIPS,
        #     with_replacement=False,
        # )
        # speedup_precision_exps(
        #     num_atoms=TRADEOFF_NUM_ATOMS,
        #     num_experiments=TRADEOFF_PCA_NUM_EXPERIMENTS,
        #     num_signals=TRADEOFF_NUM_SIGNALS,
        #     data_type=data_type,
        #     mips_alg=PCA_MIPS,
        # )
        # speedup_precision_exps(
        #     num_atoms=TRADEOFF_NUM_ATOMS,
        #     num_experiments=TRADEOFF_LSH_NUM_EXPERIMENTS,
        #     num_signals=TRADEOFF_NUM_SIGNALS,
        #     data_type=data_type,
        #     mips_alg=LSH_MIPS,
        # )

