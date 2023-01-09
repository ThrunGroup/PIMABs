import os
import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from utils.constants import (
    NORMAL_CUSTOM,
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    GREEDY_MIPS,
    PCA_MIPS,
    LSH_MIPS,

    NUM_SEEDS,
    SCALING_NUM_EXPERIMENTS,
    DEFAULT_MAXMIN,
    NORMALIZED_MAXMIN,
    DEFAULT_GREEDY_BUDGET,

    SCALING_DELTA,
    SCALING_DEPTH,
    SCALING_TOPK,
    SCALING_EPSILON,
    SCALING_NUM_HFUNC,
    SCALING_NUM_TABLES,
)

from utils.utils import get_sizes_arr
from data.get_data import get_data
from algorithms.mips_bandit import mips_bandit
from algorithms.greedy_mips import greedy_mips
from algorithms.lsh_mips import run_lsh_mips
from algorithms.pca_mips import run_pca_mips


def scaling_exp(
    epsilon: Optional[float] = SCALING_EPSILON,
    delta: Optional[float] = SCALING_DELTA,
    data_type: str = NORMAL_CUSTOM,
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
    num_seeds: int = NUM_SEEDS,
) -> float:
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
    # num_experiments determins the different sizes of d that we run
    avg_runtime = np.zeros(num_experiments)
    avg_budget = np.zeros(num_experiments)
    avg_budget_std = np.zeros(num_experiments)
    avg_accuracy = np.zeros(num_experiments)
    print(f"=> Scaling experiment for {mips_alg} with {data_type} dataset")

    # average over num_seeds results
    for seed in range(num_seeds):
        print(f"... running seed {seed}")
        rng = np.random.default_rng(seed)
        sizes_signal = get_sizes_arr(is_logspace, size_minmax, num_experiments=num_experiments)

        budgets = []
        budgets_std = []
        accuracy_list = []
        runtimes = []
        atoms, signals = get_data(
            num_atoms=num_atoms,
            len_signal=sizes_signal[-1],
            num_signals=num_signals,
            seed=seed,
            data_type=data_type,
        )
        maxmin = DEFAULT_MAXMIN

        # run the experiment for each signal size
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
                maxmin = NORMALIZED_MAXMIN  # Assume that the information on element-wise product is not given

            # bandit algorithms
            if mips_alg in [ACTION_ELIMINATION, ADAPTIVE_ACTION_ELIMINATION, MEDIAN_ELIMINATION]:
                log_info = (
                    "epsilon"
                    + str(epsilon)
                    + "_delta"
                    + str(delta)
                )
                start_time = time.time()
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
                runtime = time.time() - start_time
            elif mips_alg is GREEDY_MIPS:
                log_info = (
                    "budget"
                    + str(DEFAULT_GREEDY_BUDGET)
                )
                start_time = time.time()
                candidates_array, _, _, budget_array = greedy_mips(
                    atoms=atoms_subset,
                    signals=signals_subset,
                    budget=DEFAULT_GREEDY_BUDGET,
                    num_best_atoms=num_best_atoms,
                )
                runtime = time.time() - start_time
            elif mips_alg is PCA_MIPS:
                log_info = (
                    "depth"
                    + str(delta)
                )
                start_time = time.time()
                candidates_array, budget_array = run_pca_mips(
                    atoms=atoms_subset,
                    signals=signals_subset,
                    num_best_atoms=num_best_atoms,
                    delta=SCALING_DEPTH,
                )
                runtime = time.time() - start_time
            elif mips_alg is LSH_MIPS:
                log_info = (
                    "hfunc_"
                    + str(SCALING_NUM_HFUNC)
                    + "_tables"
                    + str(SCALING_NUM_TABLES)
                )
                start_time = time.time()
                candidates_array, budget_array = run_lsh_mips(
                    atoms=atoms_subset,
                    signals=signals_subset,
                    num_best_atoms=num_best_atoms,
                    num_hfunc=SCALING_NUM_HFUNC,
                    num_tables=SCALING_NUM_TABLES
                )
                runtime = time.time() - start_time
            else:
                raise NotImplementedError(f"{mips_alg} is not implemented")

            # store runtime, budget, budget_std, and accuracy of the experiment
            runtimes.append(runtime)
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

        avg_runtime += np.array(runtimes)
        avg_budget += np.array(budgets)
        avg_budget_std += np.array(budgets_std)
        avg_accuracy += np.array(accuracy_list)

    avg_runtime /= num_seeds
    avg_budget /= num_seeds
    avg_budget_std /= num_seeds
    avg_accuracy /= num_seeds

    # save to log files
    if is_log:
        log_dict = {
            "signal_sizes": sizes_signal,
            "budgets": avg_budget,
            "budgets_std": avg_budget_std,
            "accuracy": avg_accuracy,
            "runtime": avg_runtime,
        }
        log_df = pd.DataFrame(log_dict)
        log_dir = os.path.join("logs", "scaling") if dirname is None else dirname
        filename = os.path.join(
            log_dir,
            mips_alg
            + "_"
            "SCALING_"
            + data_type
            + "_"
            + "atoms"
            + str(num_atoms)
            + "_"
            + log_info
            + ".csv",
        )
        os.makedirs(log_dir, exist_ok=True)
        log_df.to_csv(filename, index=False)

    print(f"Mean accuracy of {mips_alg} for {data_type}: {np.mean(avg_accuracy)} \n")
    return np.mean(avg_accuracy)
