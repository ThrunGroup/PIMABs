import numpy as np
import math
from typing import Tuple
from exps.speedup_precision_exps import speedup_precision_exps
from utils.constants import (
    BATCH_SIZE,

    # datasets
    NORMAL_CUSTOM,
    ADVERSARIAL_CUSTOM,
    NETFLIX,
    MOVIE_LENS,

    # algorithms
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    GREEDY_MIPS,
    LSH_MIPS,
    PCA_MIPS,
    ASYMMETRIC,
    SYMMETRIC,

    # scaling specific constants
    TRADEOFF_NUM_EXPERIMENTS,
    TRADEOFF_NUM_ATOMS,
    TRADEOFF_NUM_SIGNALS,
    TRADEOFF_PCA_NUM_EXPERIMENTS,
    TRADEOFF_LSH_NUM_EXPERIMENTS,
    TRADEOFF_DIMENSION,
    TRADEOFF_DIMENSION_MOVIE,
    NUM_ATOMS_REAL,
    NUM_ATOMS_SIGNAL
)


def run_tradeoff():
    """
    Run speedup_precision tradeoff experiment for the two synthetic datasets (normal custom, adversarial custom) and
    the two real-world datsets (netflix and movie lens). This function is called by repro_script_python.py.
    """
    # Get data for the scaling experiments for the datasets (three synthetic datasets)
    for data_type in [NORMAL_CUSTOM, ADVERSARIAL_CUSTOM, NETFLIX, MOVIE_LENS]:
        if data_type in [NETFLIX, MOVIE_LENS]:
            num_atoms = NUM_ATOMS_REAL
            len_signals = NUM_ATOMS_SIGNAL
        else:
            num_atoms = TRADEOFF_NUM_ATOMS
            len_signals = TRADEOFF_DIMENSION

        speedup_precision_exps(
            num_atoms=num_atoms,
            num_experiments=TRADEOFF_NUM_EXPERIMENTS,
            len_signals=len_signals,
            num_signals=TRADEOFF_NUM_SIGNALS,
            data_type=data_type,
            mips_alg=ADAPTIVE_ACTION_ELIMINATION,
            with_replacement=False,
        )
        speedup_precision_exps(
            num_atoms=num_atoms,
            num_experiments=TRADEOFF_NUM_EXPERIMENTS,
            len_signals=len_signals,
            num_signals=TRADEOFF_NUM_SIGNALS,
            data_type=data_type,
            mips_alg=ACTION_ELIMINATION,
            with_replacement=False,
        )
        speedup_precision_exps(
            num_atoms=num_atoms,
            num_experiments=TRADEOFF_NUM_EXPERIMENTS,
            len_signals=len_signals,
            num_signals=TRADEOFF_NUM_SIGNALS,
            data_type=data_type,
            mips_alg=MEDIAN_ELIMINATION,
            with_replacement=False,
        )
        speedup_precision_exps(
            num_atoms=num_atoms,
            num_experiments=TRADEOFF_NUM_EXPERIMENTS,
            len_signals=len_signals,
            num_signals=TRADEOFF_NUM_SIGNALS,
            data_type=data_type,
            mips_alg=GREEDY_MIPS,
            with_replacement=False,
        )
        speedup_precision_exps(
            num_atoms=num_atoms,
            num_experiments=TRADEOFF_PCA_NUM_EXPERIMENTS,
            len_signals=len_signals,
            num_signals=TRADEOFF_NUM_SIGNALS,
            data_type=data_type,
            mips_alg=PCA_MIPS,
        )
        speedup_precision_exps(
            num_atoms=num_atoms,
            num_experiments=TRADEOFF_LSH_NUM_EXPERIMENTS,
            len_signals=len_signals,
            num_signals=TRADEOFF_NUM_SIGNALS,
            data_type=data_type,
            mips_alg=LSH_MIPS,
            lsh_type=ASYMMETRIC,
        )


if __name__ == "__main__":
   run_tradeoff()

