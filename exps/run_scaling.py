from typing import List
from exps.scaling_exp import scaling_exp
from utils.constants import (
    BATCH_SIZE,
    DEFAULT_MAXMIN,
    CRYPTO_PAIRS_MAXMIN,

    # datasets
    NORMAL_CUSTOM,
    ADVERSARIAL_CUSTOM,
    NORMAL_PAPER,
    NETFLIX,
    MOVIE_LENS,
    CRYPTO_PAIRS,

    # algorithms
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    GREEDY_MIPS,
    LSH_MIPS,
    PCA_MIPS,

    # scaling specific constants
    SCALING_NUM_EXPERIMENTS,
    SCALING_NUM_SIGNALS,
    SCALING_NUM_ATOMS,
    SCALING_NUM_ATOMS_CRYPTO_PAIRS,
    SCALING_EPSILON,
    SCALING_DELTA,
    SCALING_SIZE_MINMAX,
    SCAlING_SIZE_MINMAX_MOVIE,
    SCAlING_SIZE_MINMAX_CRYPTO_PAIRS,
    SCALING_DELTA_REAL,
    SCALING_EPSILON_REAL,
)


def run_scaling(algorithms: List[str], data_types: List[str]):
    """
    Run scaling experiment for the two synthetic datasets (normal custom, adversarial custom) and
    the two real-world datsets (netflix and movie lens). This function is called by repro_script_python.py.
    """
    # Get data for the scaling experiments for the datasets (three synthetic datasets)
    for data_type in data_types:  # NORMAL_CUSTOM, NETFLIX, MOVIE_LENS
        if data_type is CRYPTO_PAIRS:
            num_atoms = SCALING_NUM_ATOMS_CRYPTO_PAIRS
            maxmin = CRYPTO_PAIRS_MAXMIN
        else:
            num_atoms = SCALING_NUM_ATOMS
            maxmin = DEFAULT_MAXMIN

        if data_type is MOVIE_LENS:
            size_minmax = SCAlING_SIZE_MINMAX_MOVIE
        elif data_type is CRYPTO_PAIRS:
            size_minmax = SCAlING_SIZE_MINMAX_CRYPTO_PAIRS
        else:
            size_minmax = SCALING_SIZE_MINMAX

        if data_type in [NORMAL_CUSTOM, ADVERSARIAL_CUSTOM]:
            epsilon = SCALING_EPSILON
            delta = SCALING_DELTA
        elif data_type in [CRYPTO_PAIRS]:
            epsilon = SCALING_EPSILON
            delta = 0.0001
        else:  # real datasets
            epsilon = SCALING_EPSILON_REAL
            delta = SCALING_DELTA_REAL

        for algorithm in algorithms:
            parameters = {
                'epsilon': epsilon,
                'delta': delta,
                'maxmin': maxmin,
                'num_atoms': num_atoms,
                'size_minmax': size_minmax,
                'num_experiments': SCALING_NUM_EXPERIMENTS,
                'num_signals': SCALING_NUM_SIGNALS,
                'data_type': data_type,
                'with_replacement': False,
                'is_normalize': False,
                'is_logspace': True,
                'mips_alg': algorithm
            }

            scaling_exp(**parameters)


if __name__ == "__main__":
    algorithms = [
        # ADAPTIVE_ACTION_ELIMINATION,
        ACTION_ELIMINATION,
        # MEDIAN_ELIMINATION,
        # GREEDY_MIPS,
        # PCA_MIPS,
        # LSH_MIPS,
    ]

    data_types = [
        # NORMAL_CUSTOM,
        # ADVERSARIAL_CUSTOM,
        NETFLIX,
        # MOVIE_LENS,
        # CRYPTO_PAIRS,
    ]

    run_scaling(
        algorithms=algorithms, data_types=data_types)
