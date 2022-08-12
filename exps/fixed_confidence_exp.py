import numpy as np
import math
from typing import Tuple

from utils.constants import (
    BATCH_SIZE,
    ADVERSARIAL_CUSTOM,
    NORMAL_CUSTOM,
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    HOEFFDING,
    UNIFORM_PAPER,
    NORMAL_PAPER,
)
from data.custom_data import generate_custom_data

from algorithms.mips_bandit import mips_bandit

def fixed_confidence_exp(
    epsilons: Tuple[float],
    deltas: Tuple[float],
    data_type: str = ADVERSARIAL_CUSTOM,
    bandit_alg: str = MEDIAN_ELIMINATION,
    num_experiments: int = 20,
    verbose: bool = False,
    with_replacement: bool = True,
) -> None:
    """
    Validate that bandit algorithm for fixed confidence setting works well. In other words, observed failure
    probability for bandit algorithm with fixed confidence setting should be lower than failure probability(delta)
    passed as a parameter for a bandit algorithm.

    :param epsilons: A tuple of epsilons
    :param deltas: A tuple of deltas
    :param data_type: A name of data generating function
    :param bandit_alg: A name of bandit algorithm
    :param num_experiments: Number of experiments for each fixed confidence setting.
    :param verbose: verbosity of function
    :param with_replacement: Whether the algorithm mnist with replacement
    """
    for epsilon in epsilons:
        for delta in deltas:
            failure_counts = 0
            print(f"Doing an experiment with epsilon={epsilon} and delta={delta}")
            for seed in range(num_experiments):
                if verbose:
                    print(f"-seed: {seed}")
                atoms, signal = generate_custom_data(seed=seed, data_type=data_type)
                maxmin = (1, -1)
                if verbose:
                    print("--data generated")
                true_inner_products = np.matmul(atoms, signal)
                true_best_idx = np.argmax(true_inner_products)
                if verbose:
                    print("--naively solved")
                estimated_best_idx, _ = mips_bandit(
                    atoms=atoms,
                    signals=signal,
                    maxmin=maxmin,
                    bandit_alg=bandit_alg,
                    epsilon=epsilon,
                    delta=delta,
                    num_best_atoms=1,
                    seed=seed,
                    is_experiment=False,
                    with_replacement=with_replacement,
                )
                if verbose:
                    print("--stochastically solved")
                best_arm_mean = true_inner_products[true_best_idx] / signal.shape[0]
                estimated_best_arm_mean = (
                    true_inner_products[estimated_best_idx][0] / signal.shape[0]
                )
                if abs(best_arm_mean - estimated_best_arm_mean) > epsilon:
                    failure_counts += 1
            observed_delta = failure_counts / num_experiments
            print(f"delta: {delta}, epsilon: {epsilon}")
            print(f"observed failure probability: {observed_delta}")
            print(f"upper bound of failure probability: {delta}\n")
            assert (
                observed_delta < delta
            ), f"{bandit_alg} is theoretically wrong in fixed confidence setting"
