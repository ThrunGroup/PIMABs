import numpy as np
import math
from typing import Tuple
from scaling_exp import scaling_exp
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
    SCALING_NUM_EXPERIMENTS,
    SCALING_NUM_ATOMS,
    SCALING_NUM_SIGNALS,
    SCALING_SIZE_MINMAX,
)


if __name__ == "__main__":
    # Get data for the scaling experiments for the datasets (three synthetic datasets)
    #for data_type in [NORMAL_CUSTOM, NORMAL_PAPER]:
    for data_type in [ADVERSARIAL_CUSTOM]:
        # scaling_exp(
        #     num_atoms=SCALING_NUM_ATOMS,
        #     size_minmax=SCALING_SIZE_MINMAX,
        #     num_experiments=SCALING_NUM_EXPERIMENTS,
        #     num_signals=SCALING_NUM_SIGNALS,
        #     data_type=data_type,
        #     mips_alg=ADAPTIVE_ACTION_ELIMINATION,
        #     with_replacement=False,
        #     is_normalize=False,
        # )
        # scaling_exp(
        #     num_atoms=SCALING_NUM_ATOMS,
        #     size_minmax=SCALING_SIZE_MINMAX,
        #     num_experiments=SCALING_NUM_EXPERIMENTS,
        #     num_signals=SCALING_NUM_SIGNALS,
        #     data_type=data_type,
        #     mips_alg=ACTION_ELIMINATION,
        #     with_replacement=False,
        #     is_normalize=False,
        # )
        # scaling_exp(
        #     num_atoms=SCALING_NUM_ATOMS,
        #     size_minmax=SCALING_SIZE_MINMAX,
        #     num_experiments=SCALING_NUM_EXPERIMENTS,
        #     num_signals=SCALING_NUM_SIGNALS,
        #     data_type=data_type,
        #     mips_alg=MEDIAN_ELIMINATION,
        #     with_replacement=False,
        #     is_normalize=False,
        # )
        # scaling_exp(
        #     num_atoms=SCALING_NUM_ATOMS,
        #     size_minmax=SCALING_SIZE_MINMAX,
        #     num_experiments=SCALING_NUM_EXPERIMENTS,
        #     num_signals=SCALING_NUM_SIGNALS,
        #     data_type=data_type,
        #     mips_alg=GREEDY_MIPS,
        #     with_replacement=False,
        #     is_normalize=False,
        # )
        # scaling_exp(
        #     num_atoms=SCALING_NUM_ATOMS,
        #     size_minmax=SCALING_SIZE_MINMAX,
        #     num_experiments=SCALING_NUM_EXPERIMENTS,
        #     num_signals=SCALING_NUM_SIGNALS,
        #     data_type=data_type,
        #     mips_alg=PCA_MIPS,
        #     with_replacement=False,
        #     is_normalize=False,
        # )
        scaling_exp(
            num_atoms=SCALING_NUM_ATOMS,
            size_minmax=SCALING_SIZE_MINMAX,
            num_experiments=SCALING_NUM_EXPERIMENTS,
            num_signals=SCALING_NUM_SIGNALS,
            data_type=data_type,
            mips_alg=LSH_MIPS,
            with_replacement=False,
            is_normalize=False,
        )

