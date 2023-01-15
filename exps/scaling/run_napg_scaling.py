from exps.napg_scaling_exp import napg_scaling_exp
from utils.constants import (
    # datasets
    NORMAL_CUSTOM,
    ADVERSARIAL_CUSTOM,
    NETFLIX,
    MOVIE_LENS,

    # scaling specific constants
    SCALING_NUM_EXPERIMENTS,
    SCALING_NUM_ATOMS,
    SCALING_NUM_SIGNALS,
    SCALING_SIZE_MINMAX,
    SCAlING_SIZE_MINMAX_MOVIE,
)


def run_napg_scaling():
    """
    Run scaling experiment of NAPG for the two synthetic datasets (normal custom, adversarial custom) and
    the two real-world datasets (netflix and movie lens).

    This function allows the user to interrupt the experiments at any time and still have access to the previous
    results for plotting by saving the checkpoints for each seed.
    """
    # Get data for the scaling experiments for the datasets (three synthetic datasets)
    for data_type in [NORMAL_CUSTOM]:  # , ADVERSARIAL_CUSTOM, NETFLIX, MOVIE_LENS:
        if data_type is MOVIE_LENS:
            size_minmax = SCAlING_SIZE_MINMAX_MOVIE
        else:
            size_minmax = SCALING_SIZE_MINMAX
            size_minmax = (10, 100)

        napg_scaling_exp(
            num_atoms=SCALING_NUM_ATOMS,
            size_minmax=size_minmax,
            num_experiments=SCALING_NUM_EXPERIMENTS,
            num_signals=SCALING_NUM_SIGNALS,
            data_type=data_type,
            is_normalize=False,
            is_logspace=True,
        )


if __name__ == "__main__":
    run_napg_scaling()
