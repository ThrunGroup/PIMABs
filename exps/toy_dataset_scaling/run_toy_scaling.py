from exps.scaling_exp import scaling_exp
from exps.plot_baselines import create_scaling_plots
from utils.constants import (
    # datasets
    CLEAR_LEADER_HARD,
    CLEAR_LEADER_SOFT,
    NO_CLEAR_LEADER,

    # algorithms
    ACTION_ELIMINATION,

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


def run_toy_scaling(run: bool = True, plot: bool = True):
    """
    Run scaling experiments for the toy datasets
    """
    if run:
        for data_type in [CLEAR_LEADER_HARD, CLEAR_LEADER_SOFT, NO_CLEAR_LEADER]:
            # Because we expect linear scaling with d in the NO_CLEAR_LEADER dataset, we have to run it on
            # a smaller scale
            size_minmax = (10 ** 2, 10 ** 5) if data_type == NO_CLEAR_LEADER else (10 ** 2, 10 ** 6)
            mean_acc = scaling_exp(
                epsilon=0.01,
                delta=0.01,
                num_atoms=SCALING_NUM_ATOMS,
                size_minmax=size_minmax,
                num_experiments=SCALING_NUM_EXPERIMENTS,
                num_signals=SCALING_NUM_SIGNALS,
                data_type=data_type,
                mips_alg=ACTION_ELIMINATION,
                with_replacement=False,
                is_normalize=False,
                is_logspace=True,
                dir_name="toy_scaling_logs",
                num_seeds=3,
            )
            print(mean_acc)
            # assert mean_acc == 1.0, f"Mean accuracy on dataset {data_type} should be 1.0"

    if plot:
        create_scaling_plots(
            data_types=[CLEAR_LEADER_HARD, CLEAR_LEADER_SOFT, NO_CLEAR_LEADER],
            alg_names=[ACTION_ELIMINATION],
            include_error_bar=True,
            is_fit=True,
            is_logspace=True,
            dir_name="toy_scaling_logs",
        )


if __name__ == "__main__":
    run_toy_scaling(run=True, plot=True)
