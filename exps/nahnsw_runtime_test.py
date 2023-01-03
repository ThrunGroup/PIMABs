from exps.scaling_exp import scaling_exp
from utils.constants import (
    NORMAL_CUSTOM,
    NAHNSW_MIPS,

    # scaling specific constants
    SCALING_NUM_EXPERIMENTS,
    SCALING_NUM_ATOMS,
    SCALING_NUM_SIGNALS,
    SCALING_EPSILON,
    SCALING_DELTA,
    SCALING_SIZE_MINMAX,
)
import time


def run_scaling():
    # Get data for the scaling experiments for the datasets (three synthetic datasets)
    for data_type in [NORMAL_CUSTOM]:
        size_minmax = SCALING_SIZE_MINMAX
        epsilon = SCALING_EPSILON
        delta = SCALING_DELTA

        start_time = time.time()

        scaling_exp(
            epsilon=epsilon,
            delta=delta,
            num_atoms=SCALING_NUM_ATOMS,
            size_minmax=size_minmax,
            num_seeds=1,
            num_experiments=3,
            num_signals=1,
            data_type=data_type,
            mips_alg=NAHNSW_MIPS,
            with_replacement=False,
            is_normalize=False,
            is_logspace=True,
        )

        end_time = time.time()

        print("Run time: ", end_time - start_time)


if __name__ == "__main__":
    run_scaling()
