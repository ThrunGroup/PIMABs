from exps.speedup_precision_exps import speedup_precision_exps
from neq_mips.constants import TRADEOFF_NEQ_NUM_EXPERIMENTS
from utils.constants import (
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
    HNSW_MIPS,
    NAPG_MIPS,
    SYMMETRIC,
    NEQ_MIPS,

    # speedup_accuracy_tradeoff specific constants
    TRADEOFF_NUM_EXPERIMENTS,
    TRADEOFF_NUM_ATOMS,
    TRADEOFF_NUM_SIGNALS,
    TRADEOFF_PCA_NUM_EXPERIMENTS,
    TRADEOFF_LSH_NUM_EXPERIMENTS,
    TRADEOFF_DIMENSION,
    NUM_ATOMS_REAL,
    LEN_SIGNAL_NETFLIX,
    LEN_SIGNAL_MOVIE,
)


def run_tradeoff():
    """
    Run speedup_precision speedup_accuracy_tradeoff experiment for the two synthetic datasets (normal custom, adversarial custom) and
    the two real-world datasets (netflix and movie lens). This function is called by repro_script_python.py.
    """
    # Get data for the scaling experiments for the datasets (three synthetic datasets)
    for data_type in [NETFLIX, MOVIE_LENS]:  # NORMAL_CUSTOM,
        if data_type is NETFLIX:
            len_signals = LEN_SIGNAL_NETFLIX
        elif data_type is MOVIE_LENS:
            len_signals = LEN_SIGNAL_MOVIE
        else:
            len_signals = TRADEOFF_DIMENSION

        speedup_precision_exps(
            num_atoms=TRADEOFF_NUM_ATOMS,
            num_experiments=TRADEOFF_NUM_EXPERIMENTS,
            len_signals=len_signals,
            num_signals=TRADEOFF_NUM_SIGNALS,
            data_type=data_type,
            mips_alg=ADAPTIVE_ACTION_ELIMINATION,
            with_replacement=False,
        )
        speedup_precision_exps(
            num_atoms=TRADEOFF_NUM_ATOMS,
            num_experiments=TRADEOFF_NUM_EXPERIMENTS,
            len_signals=len_signals,
            num_signals=TRADEOFF_NUM_SIGNALS,
            data_type=data_type,
            mips_alg=ACTION_ELIMINATION,
            with_replacement=False,
        )
        speedup_precision_exps(
            num_atoms=TRADEOFF_NUM_ATOMS,
            num_experiments=TRADEOFF_NUM_EXPERIMENTS,
            len_signals=len_signals,
            num_signals=TRADEOFF_NUM_SIGNALS,
            data_type=data_type,
            mips_alg=MEDIAN_ELIMINATION,
            with_replacement=False,
        )
        speedup_precision_exps(
            num_atoms=TRADEOFF_NUM_ATOMS,
            num_experiments=TRADEOFF_PCA_NUM_EXPERIMENTS,
            len_signals=len_signals,
            num_signals=TRADEOFF_NUM_SIGNALS,
            data_type=data_type,
            mips_alg=PCA_MIPS,
        )
        speedup_precision_exps(
            num_atoms=TRADEOFF_NUM_ATOMS,
            num_experiments=TRADEOFF_LSH_NUM_EXPERIMENTS,
            len_signals=len_signals,
            num_signals=TRADEOFF_NUM_SIGNALS,
            data_type=data_type,
            mips_alg=LSH_MIPS,
            lsh_type=ASYMMETRIC,
        )
        speedup_precision_exps(
            num_atoms=TRADEOFF_NUM_ATOMS,
            num_experiments=TRADEOFF_NEQ_NUM_EXPERIMENTS,
            len_signals=len_signals,
            num_signals=TRADEOFF_NUM_SIGNALS,
            data_type=data_type,
            mips_alg=NEQ_MIPS,
        )
        speedup_precision_exps(
            num_atoms=TRADEOFF_NUM_ATOMS,
            num_experiments=TRADEOFF_NUM_EXPERIMENTS,
            len_signals=len_signals,
            num_signals=TRADEOFF_NUM_SIGNALS,
            data_type=data_type,
            mips_alg=NAPG_MIPS,
            with_replacement=False,
        )
        speedup_precision_exps(
            num_atoms=TRADEOFF_NUM_ATOMS,
            num_experiments=TRADEOFF_NUM_EXPERIMENTS,
            len_signals=len_signals,
            num_signals=TRADEOFF_NUM_SIGNALS,
            data_type=data_type,
            mips_alg=HNSW_MIPS,
            with_replacement=False,
        )


if __name__ == "__main__":
   run_tradeoff()
