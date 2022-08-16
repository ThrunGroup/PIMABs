import os
import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from utils.constants import (
    BATCH_SIZE,
    HOEFFDING,
    # datasets
    NORMAL_CUSTOM,
    ADVERSARIAL_CUSTOM,
    NORMAL_PAPER,
    # algorithms
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    HOEFFDING,
    GREEDY_MIPS,
    LSH_MIPS,
    PCA_MIPS,
    SYMMETRIC,
    ASYMMETRIC,
    # constants
    NUM_SEEDS,
    TRADEOFF_NUM_EXPERIMENTS,
    TRADEOFF_NUM_ATOMS,
    TRADEOFF_DIMENSION,
    TRADEOFF_NUM_SIGNALS,
    TRADEOFF_TOPK,
    TRADEOFF_PCA_NUM_EXPERIMENTS,
    TRADEOFF_DEPTH_MINMAX,
    TRADEOFF_LSH_NUM_EXPERIMENTS,
    TRADEOFF_HFUNC_MINMAX,
    TRADEOFF_TABLES_MINMAX,

    GREEDY_BUDGET_MINMAX,
    NORMAL_VAR,
    MAXMIN,
    DELTA_MINMAX,
    EPSILON_MINMAX_ACTION,
    DELTA_BINS,
    EPSILON_BINS,
    EPSILON_MINMAX_MEDIAN,
    GREEDY_BUDGET_BY,
)
from utils.utils import subset_2d, fit_and_plot
from utils.bandit_utils import get_ci
from data.custom_data import generate_custom_data

from algorithms.mips_bandit import mips_bandit
from algorithms.greedy_mips import greedy_mips, generate_conditer
from algorithms.lsh_mips import run_lsh_mips
from algorithms.pca_mips import run_pca_mips


def get_accuracy_array(
    true_candidates: np.ndarray,
    mips_candidates: np.ndarray,
) -> np.ndarray:
    """
    Returns the average accuracies of the candidates across the signals

    :true_candidates: the candidates retrieved naively
    :mip_candidates: the candidates returned by the mips algorithm
    :returns: an array of accurarices
    """
    accuracy = []
    for idx in range(TRADEOFF_NUM_SIGNALS):
        accuracy.append(len(
            np.intersect1d(true_candidates[idx], mips_candidates[idx])
        ))
    return np.array(accuracy) / (TRADEOFF_NUM_SIGNALS * TRADEOFF_TOPK)


def speedup_precision_exps(
    epsilon: Optional[float] = 0.0,
    data_type: str = NORMAL_CUSTOM,
    mips_alg: str = MEDIAN_ELIMINATION,
    len_signals: int = TRADEOFF_DIMENSION,
    num_atoms: int = TRADEOFF_NUM_ATOMS,
    num_experiments: int = TRADEOFF_NUM_EXPERIMENTS,
    num_signals: int = TRADEOFF_NUM_SIGNALS,
    num_best_atoms: int = TRADEOFF_TOPK,
    with_replacement: bool = False,
    is_log: bool = True,
    is_plot: bool = False,
    dirname: str = None,
):
    """
    Run the precision-speed tradeoff of the bandit and baseline algorithms and store them in the appropriate
    log files. Note that what we're tuning "delta" for the bandit algorithms, "budget" for greedy-mips, "depth"
    of the pca-tree for pca-mips, and "num_hfunc and num_tables" for lsh-mips.

    :param epsilon: A epsilon (an additive error allowed)
    :param data_type: A name of data generating function
    :param mips_alg: A name of algorithm
    :param num_experiments: Number of experiments
    :param len_signals: The dimension "d" of the signals
    :param num_atoms: Number of atoms
    :param num_signals: Number of signals
    :param num_best_atoms: Number of atoms we rank in the descending order of inner product
    :param with_replacement: Whether to sample with replacement
    :param is_normalize: Whether to normalize atoms and signal
    :param is_log: Whether to log the results
    :param is_plot: Whether to plot the results
    :param dirname: Directory name to log the results
    """
    avg_budget = np.zeros(num_experiments)
    avg_accuracy = np.zeros(num_experiments)
    avg_accuracy_std = np.zeros(num_experiments)

    for seed in range(NUM_SEEDS):
        print(f"... running seed {seed}")
        rng = np.random.default_rng(seed)
        atoms, signals = generate_custom_data(
            num_atoms=num_atoms,
            len_signal=len_signals,
            num_signals=num_signals,
            seed=seed,
            data_type=data_type,
        )
        maxmin = MAXMIN
        naive_candidates_array = (
            np.matmul(-atoms, signals.transpose())
            .argsort(axis=0)[:num_best_atoms]
            .transpose()
        )

        budgets = []
        accuracies = []
        accuracies_std = []
        if mips_alg is ADAPTIVE_ACTION_ELIMINATION:
            constants = np.linspace(0, NORMAL_VAR, num_experiments // DELTA_BINS)
            deltas = np.linspace(DELTA_MINMAX[0], DELTA_MINMAX[1], DELTA_BINS)
            assert len(constants) * len(deltas) == num_experiments
            for constant in constants:
                for delta in deltas:
                    new_maxmin = (maxmin[0] + constant, maxmin[1] - constant)
                    #print(f"    => (maxmin, delta) is {(new_maxmin, delta)}")
                    candidates_array, budget_array = mips_bandit(
                        atoms=atoms,
                        signals=signals,
                        maxmin=new_maxmin,
                        bandit_alg=mips_alg,
                        epsilon=epsilon,  # default 0
                        delta=delta,
                        num_best_atoms=num_best_atoms,
                        seed=seed,
                        is_experiment=False,
                        with_replacement=with_replacement,
                    )
                    budgets.append(np.mean(budget_array))
                    accuracy_array = get_accuracy_array(naive_candidates_array, candidates_array)
                    accuracies.append(np.sum(accuracy_array))
                    accuracies_std.append(np.std(accuracy_array) / np.sqrt(num_signals))

        elif mips_alg is ACTION_ELIMINATION:
            epsilons = np.linspace(EPSILON_MINMAX_ACTION[0], EPSILON_MINMAX_ACTION[1], num_experiments//EPSILON_BINS)
            deltas = np.linspace(DELTA_MINMAX[0], DELTA_MINMAX[1], EPSILON_BINS)
            for epsilon in epsilons:
                for delta in deltas:
                    #print(f"    => epsilon delta is {(epsilon, delta)}")
                    candidates_array, budget_array = mips_bandit(
                        atoms=atoms,
                        signals=signals,
                        maxmin=(-1,1),
                        bandit_alg=mips_alg,
                        epsilon=epsilon,
                        delta=delta,
                        num_best_atoms=num_best_atoms,
                        seed=seed,
                        is_experiment=False,
                        with_replacement=with_replacement,
                    )
                    budgets.append(np.mean(budget_array))
                    accuracy_array = get_accuracy_array(naive_candidates_array, candidates_array)
                    accuracies.append(np.sum(accuracy_array))
                    accuracies_std.append(np.std(accuracy_array) / np.sqrt(num_signals))

        elif mips_alg is MEDIAN_ELIMINATION:
            epsilons = np.linspace(EPSILON_MINMAX_MEDIAN[0], EPSILON_MINMAX_MEDIAN[1], num_experiments)
            #deltas = np.linspace(0.0, 0.5, 3)
            #assert len(epsilons) * len(deltas) == num_experiments

            for epsilon in epsilons:
                #for delta in deltas:
                #print(f"    => epsilon, delta is {(epsilon, delta)}")
                candidates_array, budget_array = mips_bandit(
                    atoms=atoms,
                    signals=signals,
                    maxmin=(1.0, -1.0),  # this param doesn't matter
                    bandit_alg=MEDIAN_ELIMINATION,
                    epsilon=epsilon,
                    delta=0.5,
                    num_best_atoms=num_best_atoms,
                    seed=seed,
                    is_experiment=False,
                    with_replacement=with_replacement,
                )
                budgets.append(np.mean(budget_array))
                accuracy_array = get_accuracy_array(naive_candidates_array, candidates_array)
                accuracies.append(np.sum(accuracy_array))
                accuracies_std.append(np.std(accuracy_array) / np.sqrt(num_signals))

        elif mips_alg is GREEDY_MIPS:
            conditer = generate_conditer(atoms)
            for budget in np.geomspace(2, num_atoms//GREEDY_BUDGET_BY, num_experiments):
                #print(f"    => budget is {budget}")
                candidates_array, _, _, budget_array = greedy_mips(
                    atoms=atoms,
                    signals=signals,
                    budget=int(budget),
                    num_best_atoms=num_best_atoms,
                    conditer=conditer,
                )
                budgets.append(np.mean(budget_array))
                accuracy_array = get_accuracy_array(naive_candidates_array, candidates_array)
                accuracies.append(np.sum(accuracy_array))
                accuracies_std.append(np.std(accuracy_array) / np.sqrt(num_signals))

        elif mips_alg is PCA_MIPS:
            for depth in np.linspace(
                    TRADEOFF_DEPTH_MINMAX[0], TRADEOFF_DEPTH_MINMAX[1], num_experiments
            ).astype(np.int64):
                print(f"    => depth is {depth}")
                candidates_array, budget_array = run_pca_mips(
                    atoms=atoms,
                    signals=signals,
                    num_best_atoms=num_best_atoms,
                    delta=depth,
                )
                budgets.append(np.mean(budget_array))
                accuracy_array = get_accuracy_array(naive_candidates_array, candidates_array)
                accuracies.append(np.sum(accuracy_array))
                accuracies_std.append(np.std(accuracy_array) / np.sqrt(num_signals))
        elif mips_alg is LSH_MIPS:
            table_diff = TRADEOFF_TABLES_MINMAX[1] - TRADEOFF_TABLES_MINMAX[0] + 1
            num_hfuncs = np.linspace(
                TRADEOFF_HFUNC_MINMAX[0], TRADEOFF_HFUNC_MINMAX[1], num_experiments // table_diff
            ).astype(np.int64)
            num_tables = np.linspace(
                TRADEOFF_TABLES_MINMAX[0], TRADEOFF_TABLES_MINMAX[1], table_diff
            ).astype(np.int64)
            assert len(num_hfuncs) * len(num_tables) == num_experiments
            for num_table in num_tables:
                for num_hfunc in num_hfuncs:
                    print("    => num_tables, num_hfunc: ", (num_table, num_hfunc))
                    candidates_array, budget_array = run_lsh_mips(
                        atoms=atoms,
                        signals=signals,
                        num_best_atoms=num_best_atoms,
                        num_hfunc=num_hfunc,
                        num_tables=num_table,
                        type=ASYMMETRIC
                    )
                    budgets.append(np.mean(budget_array))
                    accuracy_array = get_accuracy_array(naive_candidates_array, candidates_array)
                    accuracies.append(np.sum(accuracy_array))
                    accuracies_std.append(np.std(accuracy_array) / np.sqrt(num_signals))
        else:
            raise NotImplementedError(f"{mips_alg} is not implemented")

        avg_budget += np.array(budgets)
        avg_accuracy += np.array(accuracies)
        avg_accuracy_std += np.array(accuracies_std)

    avg_budget /= NUM_SEEDS
    avg_accuracy /= NUM_SEEDS
    avg_accuracy_std /= NUM_SEEDS

    if is_log:
        log_dict = {
            "data_type": data_type,
            "budgets": avg_budget,
            "accuracy": avg_accuracy,
            "accuracy_std": avg_accuracy_std,
        }
        log_df = pd.DataFrame(log_dict)
        log_dir = "logs/precision_speed" if dirname is None else dirname
        filename = os.path.join(
            log_dir,
            mips_alg
            + "_"
            "SPEEDUP_"
            + data_type
            + ".csv",
        )
        os.makedirs(log_dir, exist_ok=True)
        log_df.to_csv(filename, index=False)
    print(f"Finished {mips_alg} for {data_type}")

    if is_plot:
        plt.scatter(speedup_list, accuracy_list)
        plt.title(f"{mips_alg}, data size: {(num_atoms, len_signal)}")
        plt.ylabel("Accuracy")
        plt.xlabel(f"Speedup over naive approaches{naive_runtime}(s)")
        plt.show()
