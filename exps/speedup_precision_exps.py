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
    NETFLIX,
    MOVIE_LENS,

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
    HNSW_MIPS,
    NAPG_MIPS,

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
    TRADEOFF_DEFAULT_EPSILON,

    GREEDY_BUDGET_MINMAX,
    NORMAL_VAR,
    TRADEOFF_MAXMIN,
    DELTA_MINMAX,
    DELTA_MEDIAN,
    EPSILON_MINMAX_ACTION,
    DELTA_BINS,
    DELTA_BINS_REAL,
    DELTA_BINS_ACTION,
    EPSILON_MINMAX_MEDIAN,
    GREEDY_BUDGET_BY,
    NORMALIZED_MAXMIN,

    TRADEOFF_DELTA_MINMAX_REAL,
    TRADEOFF_MAXMIN_REAL,
    TRADEOFF_VAR_REAL,
    TRADEOFF_DEPTH_MINMAX_REAL,
    TRADEOFF_BATCH_SIZE_REAL,
    DELTA_POWER_MINMAX,
    DELTA_BINS_AD,
    MAXMIN_AD,
)
from data.get_data import get_data

from algorithms.mips_bandit import mips_bandit
from algorithms.greedy_mips import greedy_mips, generate_conditer
from algorithms.lsh_mips import run_lsh_mips
from algorithms.pca_mips import run_pca_mips
from algorithms.hnsw import hnsw


def get_accuracy_array(
        true_candidates: np.ndarray,
        mips_candidates: np.ndarray,
) -> np.ndarray:
    """
    Returns the average accuracy of the candidates across the signals.
    Note that this is equivalent to recall when the topk atoms is fixed.

    :param true_candidates: the candidates retrieved naively
    :param mip_candidates: the candidates returned by the mips algorithm
    :param returns: an array of accuracies
    """
    accuracy = []
    for idx in range(TRADEOFF_NUM_SIGNALS):
        accuracy.append(len(
            np.intersect1d(true_candidates[idx], mips_candidates[idx])
        ))
    return np.array(accuracy) / (TRADEOFF_NUM_SIGNALS * TRADEOFF_TOPK)


def speedup_precision_exps(
        epsilon: Optional[float] = TRADEOFF_DEFAULT_EPSILON,
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
        lsh_type: str = ASYMMETRIC,
):
    """
    Run the precision-speed tradeoff of the bandit and baseline algorithms and store them in the appropriate
    log files. Note that what we're tuning "delta" for the bandit algorithms, "budget" for greedy-mips, "depth"
    of the pca-tree for pca-mips, and "num_hfunc and num_tables" for lsh-mips.

    :param epsilon: A epsilon (an additive error allowed)
    :param data_type: A name of data generating function
    :param mips_alg: A name of algorithm
    :param len_signals: The dimension "d" of the signals
    :param num_atoms: Number of atoms
    :param num_experiments: Number of experiments
    :param num_signals: Number of signals
    :param num_best_atoms: Number of atoms we rank in the descending order of inner product
    :param with_replacement: Whether to sample with replacement
    :param is_log: Whether to log the results
    :param is_plot: Whether to plot the results
    :param dirname: Directory name to log the results
    """
    avg_runtime = np.zeros(num_experiments)
    avg_budget = np.zeros(num_experiments)
    avg_accuracy = np.zeros(num_experiments)
    avg_accuracy_std = np.zeros(num_experiments)
    print(f"=> Tradeoff experiment for {mips_alg} with {data_type} dataset")

    # setting up params
    if data_type in [NETFLIX, MOVIE_LENS]:
        tradeoff_delta_minmax = TRADEOFF_DELTA_MINMAX_REAL
        tradeoff_delta_bins = DELTA_BINS_REAL
        tradeoff_maxmin = TRADEOFF_MAXMIN_REAL
        tradeoff_var = TRADEOFF_VAR_REAL
        tradeoff_depth_minmax = TRADEOFF_DEPTH_MINMAX_REAL
        tradeoff_batch_size = TRADEOFF_BATCH_SIZE_REAL
        greedy_budget = num_atoms - 1

    elif data_type is ADVERSARIAL_CUSTOM:
        tradeoff_delta_minmax = TRADEOFF_DELTA_MINMAX_REAL
        tradeoff_delta_bins = DELTA_BINS_AD
        tradeoff_maxmin = MAXMIN_AD
        tradeoff_var = TRADEOFF_VAR_REAL
        tradeoff_depth_minmax = TRADEOFF_DEPTH_MINMAX
        tradeoff_batch_size = TRADEOFF_BATCH_SIZE_REAL
        greedy_budget = num_atoms - 1

    else:
        tradeoff_delta_minmax = DELTA_MINMAX
        tradeoff_delta_bins = DELTA_BINS
        tradeoff_maxmin = TRADEOFF_MAXMIN
        tradeoff_var = NORMAL_VAR
        tradeoff_depth_minmax = TRADEOFF_DEPTH_MINMAX
        tradeoff_batch_size = BATCH_SIZE
        greedy_budget = num_atoms // GREEDY_BUDGET_BY

    for seed in range(NUM_SEEDS):
        print(f"... running seed {seed}")
        rng = np.random.default_rng(seed)
        atoms, signals = get_data(
            num_atoms=num_atoms,
            len_signal=len_signals,
            num_signals=num_signals,
            seed=seed,
            data_type=data_type,
        )

        # these are the oracle MIPS atoms
        naive_candidates_array = (
            np.matmul(-atoms, signals.transpose())
                .argsort(axis=0)[:num_best_atoms]
                .transpose()
        )

        runtimes = []
        budgets = []
        accuracies = []
        accuracies_std = []

        if mips_alg is ADAPTIVE_ACTION_ELIMINATION or \
                (mips_alg is ACTION_ELIMINATION and data_type in [NETFLIX, MOVIE_LENS]):
            # tune bounds and deltas. "epsilon" remains the same default value
            constants = np.linspace(0, tradeoff_var, num_experiments // tradeoff_delta_bins)
            deltas = np.linspace(tradeoff_delta_minmax[0], tradeoff_delta_minmax[1], tradeoff_delta_bins)
            assert len(constants) * len(deltas) == num_experiments

            for constant in constants:
                for delta in deltas:
                    new_maxmin = (tradeoff_maxmin[0] + constant, tradeoff_maxmin[1] - constant)
                    log_info = (
                            "maxmin"
                            + str(new_maxmin)
                            + "_delta"
                            + str(delta)
                    )
                    start_time = time.time()
                    candidates_array, budget_array = mips_bandit(
                        atoms=atoms,
                        signals=signals,
                        maxmin=new_maxmin,
                        bandit_alg=mips_alg,
                        epsilon=epsilon,
                        delta=delta,
                        num_best_atoms=num_best_atoms,
                        seed=seed,
                        is_experiment=False,
                        with_replacement=with_replacement,
                        batch_size=tradeoff_batch_size,
                    )
                    runtimes.append(time.time() - start_time)
                    budgets.append(np.mean(budget_array))

                    accuracy_array = get_accuracy_array(naive_candidates_array, candidates_array)
                    accuracies.append(np.sum(accuracy_array))
                    accuracies_std.append(np.std(accuracy_array) / np.sqrt(num_signals))

        elif mips_alg is ACTION_ELIMINATION:
            epsilons = np.linspace(EPSILON_MINMAX_ACTION[0], EPSILON_MINMAX_ACTION[1],
                                   num_experiments // DELTA_BINS_ACTION)
            deltas = np.linspace(DELTA_MINMAX[0], DELTA_MINMAX[1], DELTA_BINS_ACTION)

            for epsilon in epsilons:
                for delta in deltas:
                    log_info = (
                            "epsilon"
                            + str(epsilon)
                            + "_delta"
                            + str(delta)
                    )
                    start_time = time.time()
                    candidates_array, budget_array = mips_bandit(
                        atoms=atoms,
                        signals=signals,
                        maxmin=NORMALIZED_MAXMIN,
                        bandit_alg=mips_alg,
                        epsilon=epsilon,
                        delta=delta,
                        num_best_atoms=num_best_atoms,
                        seed=seed,
                        is_experiment=False,
                        with_replacement=with_replacement,
                    )
                    runtimes.append(time.time() - start_time)
                    budgets.append(np.mean(budget_array))
                    accuracy_array = get_accuracy_array(naive_candidates_array, candidates_array)
                    accuracies.append(np.sum(accuracy_array))
                    accuracies_std.append(np.std(accuracy_array) / np.sqrt(num_signals))

        elif mips_alg is MEDIAN_ELIMINATION:
            if data_type in [NETFLIX, MOVIE_LENS]:
                idx = -1
                delta_powers = np.linspace(DELTA_POWER_MINMAX[0], DELTA_POWER_MINMAX[1], num_experiments)
                for delta_power in delta_powers:
                    idx += 1
                    epsilon = 2 + 7 * idx / num_experiments
                    log_info = (
                            "epsilon"
                            + str(epsilon)
                            + "delta_power"
                            + str(delta_power)
                    )
                    start_time = time.time()
                    candidates_array, budget_array = mips_bandit(
                        atoms=atoms,
                        signals=signals,
                        maxmin=tradeoff_maxmin,
                        bandit_alg=mips_alg,
                        epsilon=epsilon,
                        delta=1 - 10 ** (-delta_power),
                        num_best_atoms=num_best_atoms,
                        seed=seed,
                        is_experiment=False,
                        with_replacement=with_replacement,
                    )
                    runtimes.append(time.time() - start_time)
                    budgets.append(np.mean(budget_array))
                    accuracy_array = get_accuracy_array(naive_candidates_array, candidates_array)
                    accuracies.append(np.sum(accuracy_array))
                    accuracies_std.append(np.std(accuracy_array) / np.sqrt(num_signals))
            else:
                epsilons = np.linspace(EPSILON_MINMAX_MEDIAN[0], EPSILON_MINMAX_MEDIAN[1], num_experiments)
                for epsilon in epsilons:
                    log_info = (
                            "epsilon"
                            + str(epsilon)
                    )
                    start_time = time.time()
                    candidates_array, budget_array = mips_bandit(
                        atoms=atoms,
                        signals=signals,
                        maxmin=NORMALIZED_MAXMIN,  # this param doesn't matter
                        bandit_alg=MEDIAN_ELIMINATION,
                        epsilon=epsilon,
                        delta=DELTA_MEDIAN,
                        num_best_atoms=num_best_atoms,
                        seed=seed,
                        is_experiment=False,
                        with_replacement=with_replacement,
                    )
                    runtimes.append(time.time() - start_time)
                    budgets.append(np.mean(budget_array))
                    accuracy_array = get_accuracy_array(naive_candidates_array, candidates_array)
                    accuracies.append(np.sum(accuracy_array))
                    accuracies_std.append(np.std(accuracy_array) / np.sqrt(num_signals))

        elif mips_alg is GREEDY_MIPS:
            conditer = generate_conditer(atoms)

            for budget in np.geomspace(2, greedy_budget, num_experiments):
                log_info = (
                        "budget"
                        + str(budget)
                )
                start_time = time.time()
                candidates_array, _, _, budget_array = greedy_mips(
                    atoms=atoms,
                    signals=signals,
                    budget=int(budget),
                    num_best_atoms=num_best_atoms,
                    conditer=conditer,
                )
                runtimes.append(time.time() - start_time)
                budgets.append(np.mean(budget_array))
                accuracy_array = get_accuracy_array(naive_candidates_array, candidates_array)
                accuracies.append(np.sum(accuracy_array))
                accuracies_std.append(np.std(accuracy_array) / np.sqrt(num_signals))

        elif mips_alg is PCA_MIPS:
            for depth in np.linspace(
                    tradeoff_depth_minmax[0], tradeoff_depth_minmax[1], num_experiments
            ).astype(np.int64):
                log_info = (
                        "depth"
                        + str(depth)
                )
                start_time = time.time()
                candidates_array, budget_array = run_pca_mips(
                    atoms=atoms,
                    signals=signals,
                    num_best_atoms=num_best_atoms,
                    delta=depth,
                )
                runtimes.append(time.time() - start_time)
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
                    log_info = (
                            lsh_type
                            + "_"
                            + "hfunc"
                            + str(num_hfunc)
                            + "_num_table"
                            + str(num_table)
                    )
                    start_time = time.time()
                    candidates_array, budget_array = run_lsh_mips(
                        atoms=atoms,
                        signals=signals,
                        num_best_atoms=num_best_atoms,
                        num_hfunc=num_hfunc,
                        num_tables=num_table,
                        type=lsh_type
                    )
                    runtimes.append(time.time() - start_time)
                    budgets.append(np.mean(budget_array))
                    accuracy_array = get_accuracy_array(naive_candidates_array, candidates_array)
                    accuracies.append(np.sum(accuracy_array))
                    accuracies_std.append(np.std(accuracy_array) / np.sqrt(num_signals))

        elif mips_alg in [HNSW_MIPS, NAPG_MIPS]:
            Ms = np.geomspace(4, 64, num_experiments).astype(np.int64)
            ef_constructions = np.geomspace(4, num_atoms//2, num_experiments).astype(np.int64)
            ef_searches = np.geomspace(2, num_atoms//2, num_experiments).astype(np.int64)

            for (M, ef_construction, ef_search) in zip(Ms, ef_constructions, ef_searches):
                log_info = (
                        "ef_construction"
                        + str(ef_construction)
                )

                start_time = time.time()
                candidates_array, budget_array = hnsw(
                    atoms=atoms,
                    signals=signals,
                    ef_construction=ef_construction,
                    ef_search=ef_search,
                    num_links_per_node=M,
                    num_best_atoms=num_best_atoms,
                    use_norm_adjusted_factors=(mips_alg == NAPG_MIPS)
                )

                runtimes.append(time.time() - start_time)
                budgets.append(np.mean(budget_array))
                accuracy_array = get_accuracy_array(naive_candidates_array, candidates_array)
                accuracies.append(np.sum(accuracy_array))
                accuracies_std.append(np.std(accuracy_array) / np.sqrt(num_signals))

                print(ef_search, np.mean(budget_array), np.sum(accuracy_array))
        else:
            raise NotImplementedError(f"{mips_alg} is not implemented")

        avg_budget += np.array(budgets)
        avg_accuracy += np.array(accuracies)
        avg_accuracy_std += np.array(accuracies_std)
        avg_runtime += np.array(runtimes)

    avg_budget /= NUM_SEEDS
    avg_accuracy /= NUM_SEEDS
    avg_accuracy_std /= NUM_SEEDS
    avg_runtime /= NUM_SEEDS

    if is_log:
        log_dict = {
            "data_type": data_type,
            "budgets": avg_budget,
            "accuracy": avg_accuracy,
            "accuracy_std": avg_accuracy_std,
            "runtime": avg_runtime,
        }
        log_df = pd.DataFrame(log_dict)
        log_dir = os.path.join(os.getcwd(), "logs", "precision_speed") if dirname is None else dirname
        filename = os.path.join(
            log_dir,
            mips_alg
            + "_"
            + "SPEEDUP"
            + "_"
            + data_type
            + "_"
            + "atoms"
            + str(num_atoms)
            + "signal_size"
            + str(len_signals)
            + "_"
            + "topk"
            + str(num_best_atoms)
            + "_"
            + log_info
            + ".csv",
        )
        os.makedirs(log_dir, exist_ok=True)
        log_df.to_csv(filename, index=False)
    print(f"=> Finished {mips_alg} for {data_type} \n")
