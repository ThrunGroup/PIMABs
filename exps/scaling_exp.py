import os
import numpy as np
import pandas as pd
import math
from typing import Tuple, Optional
import matplotlib.pyplot as plt

from utils.constants import (
    BATCH_SIZE,
    SCALING_TOPK,
    ADVERSARIAL_CUSTOM,
    NORMAL_CUSTOM,
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    HOEFFDING,
    GREEDY_MIPS,
    NETFLIX,
    SCALING_NUM_EXPERIMENTS,
    NUM_SEEDS,

    PCA_MIPS,
    SCALING_DELTA,
    SCALING_DEPTH,

    LSH_MIPS,
    SCALING_NUM_HFUNC,
    SCALING_NUM_TABLES,
    SCALING_EPSILON,
    SCALING_DELTA,
    SYMMETRIC,
    ASYMMETRIC
)
import time

from utils.utils import subset_2d, fit_and_plot
from utils.bandit_utils import get_ci
from data.get_data import get_data
from data.custom_data import generate_custom_data
from data.get_data import get_data

from algorithms.mips_bandit import mips_bandit
from algorithms.greedy_mips import greedy_mips
from algorithms.lsh_mips import run_lsh_mips
from algorithms.pca_mips import run_pca_mips


def scaling_exp(
    epsilon: Optional[float] = SCALING_EPSILON,
    delta: Optional[float] = SCALING_DELTA,
    data_type: str = ADVERSARIAL_CUSTOM,
    mips_alg: str = MEDIAN_ELIMINATION,
    num_experiments: int = SCALING_NUM_EXPERIMENTS,
    size_minmax: Tuple[int, int] = (10, 10**6),
    num_atoms: int = 10**3,
    num_signals: int = 1,
    num_best_atoms: int = SCALING_TOPK,
    with_replacement: bool = False,
    is_normalize: bool = False,
    is_log: bool = True,
    is_plot: bool = True,
    is_logspace: bool = False,
    dirname: str = None,
) -> None:
    """
    Run the time complexity of the bandit and baseline algorithms as the signal size scales and store them in
    the appropriate log files. MEDIAN_ELIMINATION algorithm should satisfy this scaling according to Corollary 3 in
    "https://arxiv.org/pdf/1812.06360.pdf#page=5".

    :param epsilon: A epsilon (an additive error allowed)
    :param delta: A delta (an upper bound of failure probability)
    :param data_type: A name of data generating function
    :param mips_alg: A name of algorithm
    :param num_experiments: Number of experiments
    :param size_minmax: Minimum and Maximum of signal vector size
    :param num_atoms: Number of atoms
    :param num_signals: Number of signals
    :param num_best_atoms: Number of atoms we rank in the descending order of inner product
    :param with_replacement: Whether to sample with replacement
    :param is_normalize: Whether to normalize atoms and signal
    :param is_log: Whether to log the results
    :param is_plot: Whether to plot the results
    :param is_logspace: Whether to construct sizes of signal spaced evenly on a log scale
    :param dirname: Directory name to log the results
    """
    global budget_array
    # average NUM_SEEDS results
    avg_budget = np.zeros(num_experiments)
    avg_budget_std = np.zeros(num_experiments)
    avg_accuracy = np.zeros(num_experiments)
    for seed in range(NUM_SEEDS):
        print(f"... running seed {seed}")
        rng = np.random.default_rng(seed)

        if is_logspace:
            sizes_signal = np.logspace(
                np.log(size_minmax[0]), np.log(size_minmax[1]), num_experiments, base=np.e
            ).astype(np.int64)
        else:
            sizes_signal = np.linspace(
                size_minmax[0], size_minmax[1], num_experiments
            ).astype(np.int64)

        budgets = []
        budgets_std = []
        accuracy_list = []
        atoms, signals = get_data(
            num_atoms=num_atoms,
            len_signal=sizes_signal[-1],
            num_signals=num_signals,
            seed=seed,
            data_type=data_type,
        )
        maxmin = (1, -1)

        for size in sizes_signal:
            #print(f"    => signal size {size}")
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
                log_info = (str(epsilon) + "_" + str(delta))
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
                log_info = "50"
                candidates_array, _, _, budget_array = greedy_mips(
                    atoms=atoms_subset,
                    signals=signals_subset,
                    budget=50,
                    num_best_atoms=num_best_atoms,
                )
            elif mips_alg is PCA_MIPS:
                log_info = str(delta)
                candidates_array, budget_array = run_pca_mips(
                    atoms=atoms_subset,
                    signals=signals_subset,
                    num_best_atoms=num_best_atoms,
                    delta=SCALING_DEPTH,
                )
            elif mips_alg is LSH_MIPS:
                log_info = (str(SCALING_NUM_HFUNC) + "_" + str(SCALING_NUM_TABLES))
                candidates_array, budget_array = run_lsh_mips(
                    atoms=atoms_subset,
                    signals=signals_subset,
                    num_best_atoms=num_best_atoms,
                    num_hfunc=SCALING_NUM_HFUNC,
                    num_tables=SCALING_NUM_TABLES,
                    type=ASYMMETRIC
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

        avg_budget += np.array(budgets)
        avg_budget_std += np.array(budgets_std)
        avg_accuracy += np.array(accuracy_list)

    avg_budget /= NUM_SEEDS
    avg_budget_std /= NUM_SEEDS
    avg_accuracy /= NUM_SEEDS

    if is_log:
        log_dict = {
            "signal_sizes": sizes_signal,
            "budgets": avg_budget,
            "budgets_std": avg_budget_std,
            "accuracy": avg_accuracy,
        }
        log_df = pd.DataFrame(log_dict)
        log_dir = "logs/scaling" if dirname is None else dirname
        filename = os.path.join(
            log_dir,
            mips_alg
            + "_"
            "SCALING_"
            + data_type
            + ":"
            + str(size_minmax)
            + "_"
            + str(num_atoms)
            + "_"
            + str(epsilon)
            + "_"
            + str(delta)
            + "_"
            + log_info
            + ".csv",
        )
        os.makedirs(log_dir, exist_ok=True)
        log_df.to_csv(filename, index=False)

    if is_plot:
        if is_logspace:
            def power10(array):
                return np.power(10, array)

            def power10_sqrt(array):
                return np.power(10, array) ** (1 / 2)

            functions = [np.array, power10_sqrt, power10]
            function_names = [
                "$y \propto log x$",
                "$y \propto x^{1/2}$",
                "$y \propto x$",
            ]
            sizes_signal = np.log10(sizes_signal)
            xlabel = "signal sizes"
        else:
            functions = [np.array, np.sqrt, np.log]
            function_names = ["Linear fit", "Sqrt fit", "Logarithm fit"]
            xlabel = None

    print(f"Mean accuracy with {mips_alg} for {data_type}: {np.mean(avg_accuracy)}")
