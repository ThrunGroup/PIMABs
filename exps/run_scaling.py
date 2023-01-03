from typing import List
from exps.scaling_exp import scaling_exp
from utils.constants import (
    # datasets
    NORMAL_CUSTOM,
    ADVERSARIAL_CUSTOM,
    NORMAL_PAPER,
    NETFLIX,
    MOVIE_LENS,

    # algorithms
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    GREEDY_MIPS,
    LSH_MIPS,
    PCA_MIPS,
    HNSW_MIPS,
    NAHNSW_MIPS,

    # scaling specific constants
    SCALING_NUM_EXPERIMENTS,
    SCALING_NUM_ATOMS,
    SCALING_NUM_SIGNALS,
    SCALING_EPSILON,
    SCALING_DELTA,
    SCALING_SIZE_MINMAX,
    SCAlING_SIZE_MINMAX_MOVIE,
    SCALING_DELTA_REAL,
    SCALING_EPSILON_REAL,
)


def run_scaling(algorithms: List[str], datasets: List[str]):
    """
    Run scaling experiment for the two synthetic datasets (normal custom, adversarial custom) and
    the two real-world datsets (netflix and movie lens). This function is called by repro_script_python.py.
    """
    # Get data for the scaling experiments for the datasets (three synthetic datasets)
    for data_type in [NORMAL_CUSTOM, ADVERSARIAL_CUSTOM, NETFLIX, MOVIE_LENS]:
        if data_type is MOVIE_LENS:
            size_minmax = SCAlING_SIZE_MINMAX_MOVIE
        else:
            size_minmax = SCALING_SIZE_MINMAX

        if data_type in [NORMAL_CUSTOM, ADVERSARIAL_CUSTOM]:
            epsilon = SCALING_EPSILON
            delta = SCALING_DELTA
        else:  # real datasets
            epsilon = SCALING_EPSILON_REAL
            delta = SCALING_DELTA_REAL

        for algorithm in algorithms:
            parameters = {
                "epsilon": epsilon,
                "delta": delta,
                "num_atoms": SCALING_NUM_ATOMS,
                "size_minmax": size_minmax,
                "num_experiments": SCALING_NUM_EXPERIMENTS,
                "num_signals": SCALING_NUM_SIGNALS,
                "data_type": data_type,
                "mips_alg": algorithm,
                "with_replacement": False,
                "is_normalize": False,
                "is_logspace": True,
            }

            scaling_exp(**parameters)


if __name__ == "__main__":
    algorithms = [
        # ADAPTIVE_ACTION_ELIMINATION,
        # ACTION_ELIMINATION,
        # MEDIAN_ELIMINATION,
        NAHNSW_MIPS,
        HNSW_MIPS,
        # PCA_MIPS,
        # LSH_MIPS
    ]

    datasets = [
        NORMAL_CUSTOM,
        # ADVERSARIAL_CUSTOM,
        # NETFLIX,
        # MOVIE_LENS
    ]

    run_scaling(algorithms=algorithms, datasets=datasets)
