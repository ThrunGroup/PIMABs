import os
import numpy as np
import pandas as pd
import math
from typing import Tuple, Optional

from utils.constants import (
    BATCH_SIZE,
    ADVERSARIAL_CUSTOM,
    NORMAL_CUSTOM,
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    HOEFFDING,
    GREEDY_MIPS,
)
from utils.utils import subset_2d, fit_and_plot
from utils.bandit_utils import get_ci
from data.custom_data import generate_custom_data

from algorithms.mips_bandit import mips_bandit
from algorithms.greedy_mips import greedy_mips

def scaling_exp(
    epsilon: Optional[float],
    delta: Optional[float],
    data_type: str = ADVERSARIAL_CUSTOM,
    mips_alg: str = MEDIAN_ELIMINATION,
    num_experiments: int = 20,
    size_minmax: Tuple[int, int] = (10, 10 ** 6),
    num_atoms: int = 10 ** 3,
    num_signals: int = 1,
    num_best_atoms: int = 1,
    seed: int = 0,
    with_replacement: bool = False,
    is_normalize: bool = False,
    is_log: bool = True,
    is_plot: bool = True,
    is_logspace: bool = True,
    dirname: str = None,
) -> None:
    """
    Check that the time complexity of bandit algorithm scales with the square root of the signal size.
    MEDIAN_ELIMINATION algorithm should satisfy this scaling according to Corollary 3 in
    "https://arxiv.org/pdf/1812.06360.pdf#page=5".

    :param epsilon: A epsilon (an additive error allowed)
    :param delta: A delta (an upper bound of failure probability)
    :param data_type: A name of data generating function
    :param mips_alg: A name of bandit algorithm
    :param num_experiments: Number of experiments
    :param size_minmax: Minimum and Maximum of signal vector size
    :param num_atoms: Number of atoms
    :param num_signals: Number of signals
    :param num_best_atoms: Number of atoms we rank in the descending order of inner product
    :param seed: A random seed
    :param with_replacement: Whether to sample with replacement
    :param is_normalize: Whether to normalize atoms and signal
    :param is_log: Whether to log the results
    :param is_plot: Whether to plot the results
    :param is_logspace: Whether to construct sizes of signal spaced evenly on a log scale
    :param dirname: Directory name to log the results
    """
    global budget_array
    rng = np.random.default_rng(seed)
    if is_logspace:
        sizes_signal = np.logspace(np.log(size_minmax[0]), np.log(size_minmax[1]), num_experiments, base=np.e).astype(
            np.int64
        )
    else:
        sizes_signal = np.linspace(size_minmax[0], size_minmax[1], num_experiments).astype(
            np.int64
        )
    budgets = []
    budgets_std = []
    accuracy_list = []
    atoms, signals = generate_custom_data(
        num_atoms=num_atoms,
        len_signal=sizes_signal[-1],
        num_signals=num_signals,
        seed=seed,
        data_type=data_type,
    )
    maxmin = (1, -1)
    for size in sizes_signal:
        subset_indices = rng.choice(sizes_signal[-1], size, replace=False)
        atoms_subset, signals_subset = (
            atoms[:, subset_indices],
            signals[:, subset_indices],
        )
        if is_normalize:
            atoms_norm = np.linalg.norm(atoms_subset, axis=1, keepdims=True)
            signal_norm = np.linalg.norm(signals_subset, axis=1, keepdims=True)
            atoms_subset /= atoms_norm
            signals_subset /= signal_norm
            epsilon /= atoms_norm.max() * signal_norm
            maxmin = (
                1,
                -1,
            )  # Assume that the information on element-wise product is not given.
        if "ELIMINATION" in mips_alg:
            candidates_array, budget_array = mips_bandit(
                atoms=atoms_subset,
                signals=signals_subset,
                maxmin=maxmin,
                bandit_alg=mips_alg,
                epsilon=epsilon,
                delta=delta,
                num_best_atoms=num_best_atoms,
                seed=seed,
                is_experiment=False,
                with_replacement=with_replacement,
            )
        elif mips_alg is GREEDY_MIPS:
            candidates_array, _, _, budget_array = greedy_mips(
                atoms=atoms_subset,
                signals=signals_subset,
                budget=50,
                num_best_atoms=num_best_atoms,
            )
        else:
            raise NotImplementedError(f"{mips_alg} is not implemented")
        budgets.append(np.mean(budget_array))
        budgets_std.append(np.std(budget_array) / np.sqrt(num_signals))
        naive_candidates_array = (
            np.matmul(-atoms_subset, signals_subset.transpose())
            .argsort(axis=0)[:num_best_atoms]
            .transpose()
        )
        accuracy = 0
        for idx in range(num_signals):
            accuracy += len(
                np.intersect1d(naive_candidates_array[idx], candidates_array[idx])
            )
        accuracy /= num_signals * num_best_atoms
        accuracy_list.append(accuracy)

    budgets = np.array(budgets)
    budgets_std = np.array(budgets_std)
    accuracy_list = np.array(accuracy_list)

    if is_log:
        log_dict = {
            "signal_sizes": sizes_signal,
            "budgets": budgets,
            "budgets_std": budgets_std,
            "accuracy": accuracy_list,
        }
        log_df = pd.DataFrame(log_dict)
        log_dir = "logs" if dirname is None else dirname
        filename = os.path.join(
            log_dir,
            "SCALING_" + data_type + "_" + mips_alg + "_" + str(epsilon) + "_" + str(delta),
        )
        os.makedirs(log_dir, exist_ok=True)
        log_df.to_csv(filename, index=False)
    if is_plot:
        if is_logspace:
            functions = [np.array, np.exp]
            function_names = ["Linear", "Exp"]
            sizes_signal = np.log(sizes_signal)
            xlabel = "signal sizes (natural log scale)"
        else:
            functions = [np.array, np.sqrt, np.log]
            function_names = ["Linear", "Sqrt", "Logarithm"]
            xlabel = None
        fit_and_plot(
            sizes_signal,
            budgets,
            functions,
            function_names,
            ys_std=budgets_std,
            title=f"{mips_alg} + w/ replacement: {with_replacement}",
            xlabel=xlabel
        )

    print(f"Mean accuracy with {mips_alg} for {data_type}: {np.mean(accuracy_list)}")
