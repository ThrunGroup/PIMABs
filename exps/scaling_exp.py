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
)
from utils.utils import subset_2d, fit_and_plot
from utils.bandit_utils import get_ci
from data.custom_data import generate_normal_data, generate_adversarial_data

from algorithms.mips_bandit import mips_bandit
from algorithms.median_elimination import median_elimination
from algorithms.action_elimination import action_elimination
from algorithms.adaptive_action_elimination import adaptive_action_elimination

def scaling_exp(
    epsilon: float,
    delta: float,
    generate_data: str = ADVERSARIAL_CUSTOM,
    bandit_alg: str = MEDIAN_ELIMINATION,
    num_experiments: int = 20,
    size_minmax: Tuple[int] = (10, 10 ** 6),
    seed: int = 0,
    with_replacement: bool = False,
    is_normalize: bool = True,
) -> None:
    """
    Check that the time complexity of bandit algorithm scales with the square root of the signal size.
    MEDIAN_ELIMINATION algorithm should satisfy this scaling according to Corollary 3 in "shorturl.at/adPR3".

    :param epsilon: A epsilon (an additive error allowed)
    :param delta: A delta (an upper bound of failure probability)
    :param generate_data: A name of data generating function
    :param bandit_alg: A name of bandit algorithm
    :param num_experiments: Number of experiments
    :param size_minmax: Minimum and Maximum of signal vector size
    :param seed: A random seed
    :param with_replacement: Whether to sample with replacement
    :param is_normalize: Whether to normalize atoms and signal
    """
    rng = np.random.default_rng(seed)
    sizes_signal = np.linspace(size_minmax[0], size_minmax[1], num_experiments).astype(
        np.int64
    )
    budgets = []
    if generate_data == ADVERSARIAL_CUSTOM:
        generate_data = generate_adversarial_data
        maxmin = (1, 0)
    elif generate_data == NORMAL_CUSTOM:
        generate_data = generate_normal_data
        maxmin = (2, -2)
    else:
        raise NotImplementedError(f"{generate_data} is not implemented")

    atoms, signal = generate_data(len_signal=sizes_signal[-1], seed=seed)
    for size in sizes_signal:
        subset_indices = rng.choice(sizes_signal[-1], size, replace=False)
        atoms_subset, signal_subset = atoms[:, subset_indices], signal[subset_indices]
        if is_normalize:
            atoms_norm = np.linalg.norm(atoms_subset, axis=1, keepdims=True)
            signal_norm = np.linalg.norm(signal_subset, keepdims=True)
            atoms_subset /= atoms_norm
            signal_subset /= signal_norm
            epsilon /= atoms_norm.max() * signal_norm
            maxmin = (
                1,
                -1,
            )  # Assume that the information on element-wise product is not given.
        _, budget = mips_bandit(
            atoms=atoms_subset,
            signal=signal_subset,
            maxmin=maxmin,
            bandit_alg=bandit_alg,
            epsilon=epsilon,
            delta=delta,
            num_best_atoms=1,
            seed=seed,
            is_experiment=False,
            with_replacement=with_replacement,
        )
        budgets.append(budget)
    budgets = np.array(budgets)
    functions = [np.array, np.sqrt, np.log]
    function_names = ["Linear", "Sqrt", "Logarithm"]
    fit_and_plot(sizes_signal, budgets, functions, function_names)
