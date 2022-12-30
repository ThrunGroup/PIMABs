from typing import List
from exps.speedup_precision_exps import speedup_precision_exps
from utils.constants import (
    BATCH_SIZE,

    # datasets
    NORMAL_CUSTOM,
    ADVERSARIAL_CUSTOM,
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
    ASYMMETRIC,

    # scaling specific constants
    TRADEOFF_NUM_EXPERIMENTS,
    TRADEOFF_NUM_ATOMS,
    TRADEOFF_NUM_SIGNALS,
    TRADEOFF_PCA_NUM_EXPERIMENTS,
    TRADEOFF_LSH_NUM_EXPERIMENTS,
    TRADEOFF_DIMENSION,
    NUM_ATOMS_REAL,
    NUM_ATOMS_SIGNAL,
    CRYPTO_PAIRS_SIGNAL_LENGTH,
)


def run_tradeoff(algorithms: List[str]):
    """
    Run speedup_precision tradeoff experiment for the two synthetic datasets (normal custom, adversarial custom) and
    the two real-world datsets (netflix and movie lens). This function is called by repro_script_python.py.
    """

    # Get data for the scaling experiments for the datasets (three synthetic datasets)
    for data_type in [CRYPTO_PAIRS]:  # ADVERSARIAL_CUSTOM, NORMAL_CUSTOM, ADVERSARIAL_CUSTOM, NETFLIX, MOVIE_LENS
        if data_type in [CRYPTO_PAIRS, NETFLIX, MOVIE_LENS]:
            num_atoms = NUM_ATOMS_REAL
            len_signals = NUM_ATOMS_SIGNAL
            if data_type is CRYPTO_PAIRS:
                len_signals = CRYPTO_PAIRS_SIGNAL_LENGTH
        else:
            num_atoms = TRADEOFF_NUM_ATOMS
            len_signals = TRADEOFF_DIMENSION

        for algorithm in algorithms:
            parameters = {
                'num_atoms': num_atoms,
                'num_experiments': TRADEOFF_NUM_EXPERIMENTS,
                'len_signals': len_signals,
                'num_signals': TRADEOFF_NUM_SIGNALS,
                'data_type': data_type,
                'mips_alg': algorithm
            }

            if algorithm in [ADAPTIVE_ACTION_ELIMINATION, ACTION_ELIMINATION, MEDIAN_ELIMINATION, GREEDY_MIPS]:
                parameters['with_replacement'] = False

            if algorithm is LSH_MIPS:
                parameters['lsh_type'] = ASYMMETRIC

            speedup_precision_exps(**parameters)


if __name__ == "__main__":
    run_tradeoff([
        ADAPTIVE_ACTION_ELIMINATION,
        ACTION_ELIMINATION,
        # MEDIAN_ELIMINATION,
        # GREEDY_MIPS,
        # PCA_MIPS,
        # LSH_MIPS,
    ])
