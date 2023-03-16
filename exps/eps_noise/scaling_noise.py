import numpy as np
from exps.core_scaling.scaling_baselines import get_scaling_baseline_params
from exps.scaling_exp import scaling_exp
from utils.constants import (
    NETFLIX,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    SCALING_NUM_ATOMS,
    SCALING_NUM_EXPERIMENTS,
    SCALING_NUM_SIGNALS,
)


def scaling_noise(algorithms):
    """
    See if the scaling behavior for the MIPS algorithms is constant even for the noise-induced Netflix dataset.
    By default, the algorithm tested is BanditMIPS.
    """
    # Get data for the scaling experiments for the datasets (not the high-dimensional datasets)
    print(f"=> Creating Scaling log files for {algorithms}")
    for algorithm in algorithms:
        epsilon, delta, maxmin, size_minmax = get_scaling_baseline_params(NETFLIX)
        scaling_exp(
            epsilon=epsilon,
            delta=delta,
            maxmin=maxmin,
            num_atoms=SCALING_NUM_ATOMS,
            size_minmax=size_minmax,
            num_experiments=SCALING_NUM_EXPERIMENTS,
            num_signals=SCALING_NUM_SIGNALS,
            data_type=NETFLIX,
            add_noise=True,  # note this parameter
            mips_alg=algorithm,
            with_replacement=False,
            is_normalize=False,
            is_logspace=False,
        )


if __name__ == "__main__":
    scaling_noise([ACTION_ELIMINATION])