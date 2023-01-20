from typing import List

from exps.speedup_precision_exps import speedup_precision_exps
from utils.constants import (
    BATCH_SIZE,

    # datasets
    NORMAL_CUSTOM,
    UNIFORM_PAPER,
    NETFLIX,
    MOVIE_LENS,
    CRYPTO_PAIRS,
    COR_NORMAL_CUSTOM,
    CLEAR_LEADER_HARD,
    NO_CLEAR_LEADER,

    # algorithms
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    GREEDY_MIPS,
    LSH_MIPS,
    PCA_MIPS,
    HNSW_MIPS,
    NAPG_MIPS,
    NEQ_MIPS,
    BUCKET_ACTION_ELIMINATION,
    H2ALSH,
    ASYMMETRIC,

    # tradeoff specific constants
    TRADEOFF_NUM_EXPERIMENTS,
    TRADEOFF_NUM_ATOMS,
    TRADEOFF_NUM_SIGNALS,
    TRADEOFF_PCA_NUM_EXPERIMENTS,
    TRADEOFF_LSH_NUM_EXPERIMENTS,
    TRADEOFF_DIMENSION,
    NUM_ATOMS_REAL,
    LEN_SIGNAL_NETFLIX,
    LEN_SIGNAL_MOVIE,
    CRYPTO_PAIRS_SIGNAL_LENGTH,
)


def get_tradeoff_params(data_type):
    """
    Function gives the constants for [num_atoms, len_signals] for the specified datatype.
    """
    if data_type in [NETFLIX, MOVIE_LENS, CRYPTO_PAIRS]:
        num_atoms = NUM_ATOMS_REAL
    else:
        num_atoms = TRADEOFF_NUM_ATOMS

    if data_type is NETFLIX:
        len_signals = LEN_SIGNAL_NETFLIX
    elif data_type is MOVIE_LENS:
        len_signals = LEN_SIGNAL_MOVIE
    elif data_type is CRYPTO_PAIRS:
        len_signals = CRYPTO_PAIRS_SIGNAL_LENGTH
    else:
        len_signals = TRADEOFF_DIMENSION

    return num_atoms, len_signals


def tradeoff_baselines():
    """
    Todo @jey and @luke: fill up this docstring with the corresponding experiments.
    This is just an example of how I think the organization should be. I've only included the configurations for
    5 datasets (no high-dimensional ones) for 8 mips algorithsm. For each of your experiments, create a function
    that defines the configs similar to this one -- MAKE SURE TO NAME IT SOMETHING SPECIFIC -- and put it in
    this directory with the appropriate name.
    """
    # Get data for the scaling experiments for the datasets (three synthetic datasets)
    for data_type in [NORMAL_CUSTOM, UNIFORM_PAPER, CLEAR_LEADER_HARD, NETFLIX, MOVIE_LENS]:
        for algorithm in [
            GREEDY_MIPS,  # greedy approach
            LSH_MIPS,  # mapping to higher dimensions
            PCA_MIPS, NEQ_MIPS, NAPG_MIPS,  # mapping to lower dimensions
            MEDIAN_ELIMINATION, ACTION_ELIMINATION, ADAPTIVE_ACTION_ELIMINATION,  # bandit approach
        ]:
            num_atoms, len_signals = get_tradeoff_params(data_type)
            parameters = {
                'num_atoms': num_atoms,
                'num_experiments': TRADEOFF_NUM_EXPERIMENTS,
                'len_signals': len_signals,
                'num_signals': TRADEOFF_NUM_SIGNALS,
                'data_type': data_type,
                'mips_alg': algorithm,
            }

            if algorithm in [ADAPTIVE_ACTION_ELIMINATION, ACTION_ELIMINATION, MEDIAN_ELIMINATION, GREEDY_MIPS]:
                parameters['with_replacement'] = False

            if algorithm is LSH_MIPS:
                parameters['lsh_type'] = ASYMMETRIC
            speedup_precision_exps(**parameters)


if __name__ == "__main__":
    tradeoff_baselines()
