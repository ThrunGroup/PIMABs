from napg_speedup_precision_exps import napg_speedup_precision_exps
from utils.constants import (
    # datasets
    NORMAL_CUSTOM,
    ADVERSARIAL_CUSTOM,
    NETFLIX,
    MOVIE_LENS,

    # scaling specific constants
    TRADEOFF_NUM_EXPERIMENTS,
    TRADEOFF_NUM_ATOMS,
    TRADEOFF_NUM_SIGNALS,
    TRADEOFF_DIMENSION,
    NUM_ATOMS_REAL,
    LEN_SIGNAL_NETFLIX,
    LEN_SIGNAL_MOVIE,
)


def run_napg_tradeoff():
    """
    Run speedup_precision tradeoff experiment for the two synthetic datasets (normal custom, adversarial custom) and
    the two real-world datasets (netflix and movie lens). This function is called by repro_script_python.py.
    """
    # Get data for the scaling experiments for the datasets (three synthetic datasets)
    for data_type in [NORMAL_CUSTOM, ADVERSARIAL_CUSTOM, NETFLIX, MOVIE_LENS]:
        if data_type is NETFLIX:
            num_atoms = NUM_ATOMS_REAL
            len_signals = LEN_SIGNAL_NETFLIX
        elif data_type is MOVIE_LENS:
            num_atoms = NUM_ATOMS_REAL
            len_signals = LEN_SIGNAL_MOVIE
        else:
            num_atoms = TRADEOFF_NUM_ATOMS
            len_signals = TRADEOFF_DIMENSION

        napg_speedup_precision_exps(
            num_atoms=num_atoms,
            num_experiments=TRADEOFF_NUM_EXPERIMENTS,
            len_signals=len_signals,
            num_signals=TRADEOFF_NUM_SIGNALS,
            data_type=data_type,
        )


if __name__ == "__main__":
    run_napg_tradeoff()
