import os
import numpy as np
import pandas as pd
import time
from typing import Tuple

from numpy import ndarray
from pandas import read_csv
import glob

from utils.constants import (
    # Dataset
    NORMAL_CUSTOM,

    # Algorithm
    NAPG_MIPS,

    # Configurations
    NUM_SEEDS,
    SCALING_NUM_EXPERIMENTS,
    SCALING_TOPK,
)

from utils.utils import get_sizes_arr
from data.get_data import get_data
from algorithms.napg import hnsw


def napg_scaling_exp(
        data_type: str = NORMAL_CUSTOM,
        num_experiments: int = SCALING_NUM_EXPERIMENTS,
        num_seeds: int = NUM_SEEDS,
        data_generation_seed: int = 0,
        size_minmax: Tuple[int, int] = (10, 10 ** 6),
        num_atoms: int = 10 ** 3,
        num_signals: int = 1,
        num_best_atoms: int = SCALING_TOPK,
        is_normalize: bool = False,
        is_log: bool = True,
        is_logspace: bool = False,
        dirname: str = None,
) -> ndarray:
    """
    Run the time complexity of the bandit and baseline algorithms as the signal size scales and store them in
    the appropriate log files. MEDIAN_ELIMINATION algorithm should satisfy this scaling according to Corollary 3 in
    "https://arxiv.org/pdf/1812.06360.pdf#page=5".

    :param data_type: A name of data generating function
    :param num_experiments: Number of experiments
    :param num_seeds: Number of seeds
    :param data_generation_seed: Random seed for generating data
    :param size_minmax: Minimum and Maximum of signal vector size
    :param num_atoms: Number of atoms
    :param num_signals: Number of signals
    :param num_best_atoms: Number of atoms we rank in the descending order of inner product
    :param is_normalize: Whether to normalize atoms and signal
    :param is_log: Whether to log the results
    :param is_logspace: Whether to construct sizes of signal spaced evenly on a log scale
    :param dirname: Directory name to log the results
    """
    global budget_array
    # num_experiments determines the different sizes of d that we run
    print(f"=> Scaling experiment for NAPG with {data_type} dataset")

    sizes_signal = get_sizes_arr(is_logspace, size_minmax, num_experiments=num_experiments)

    # If the dataset is a toy dataset, the randomness will include the data generation process
    # If the dataset is a real dataset, the randomness will only include which vector is chosen as signal
    atoms, signals = get_data(
        num_atoms=num_atoms,
        len_signal=sizes_signal[-1],
        num_signals=num_signals,
        seed=data_generation_seed,
        data_type=data_type,
    )

    checkpoint_path = os.path.join(os.getcwd(), "logs", "scaling", "napg_checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_files = glob.glob(os.path.join(checkpoint_path, f"*{data_type}*"))
    previous_seed = len(checkpoint_files)

    # average over num_seeds results
    for seed in range(previous_seed, num_seeds):
        print(f"... running seed {seed}")
        rng = np.random.default_rng(seed)

        budgets = []
        budgets_std = []
        accuracy_list = []
        runtimes = []

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

            start_time = time.time()
            candidates_array, budget_array = hnsw(
                atoms=atoms_subset,
                signals=signals_subset,
                num_best_atoms=num_best_atoms,
                use_norm_adjusted_factors=True
            )
            runtime = time.time() - start_time

            # store runtime, budget, budget_std, and accuracy of the experiment
            runtimes.append(runtime)
            budgets.append(np.mean(budget_array))
            budgets_std.append(np.std(budget_array) / np.sqrt(num_signals))
            naive_candidates_array = np.sort(
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

        # Save the experiment results for each seed as a checkpoint
        checkpoint_log_dict = {
            "signal_sizes": sizes_signal,
            "budgets": np.array(budgets),
            "budgets_std": np.array(budgets_std),
            "accuracy": np.array(accuracy_list),
            "runtime": np.array(runtimes),
        }

        checkpoint_log_df = pd.DataFrame(checkpoint_log_dict)
        filename = os.path.join(
            checkpoint_path,
            NAPG_MIPS
            + "_"
            + data_type
            + "_"
            + "seed"
            + str(seed)
            + ".csv",
        )
        checkpoint_log_df.to_csv(filename, index=False)

        print(f"Checkpoint for seed {seed} saved")

        # Combine the previous checkpoint data and save it in a single CSV file. This allows the user to interrupt the
        # experiments at any time and still have access to the results for plotting.
        new_checkpoint_files = glob.glob(os.path.join(checkpoint_path, f"*{data_type}*"))
        num_checkpoints = len(new_checkpoint_files)

        avg_runtime = np.zeros(num_experiments)
        avg_budget = np.zeros(num_experiments)
        avg_budget_std = np.zeros(num_experiments)
        avg_accuracy = np.zeros(num_experiments)

        for checkpoint_file in new_checkpoint_files:
            data = read_csv(checkpoint_file)

            budgets = data["budgets"].tolist()
            budgets_std = data["budgets_std"].tolist()
            accuracy = data["accuracy"].tolist()
            runtime = data["runtime"].tolist()

            avg_budget += budgets
            avg_budget_std += budgets_std
            avg_accuracy += accuracy
            avg_runtime += runtime

        avg_runtime /= num_checkpoints
        avg_budget /= num_checkpoints
        avg_budget_std /= num_checkpoints
        avg_accuracy /= num_checkpoints

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
                NAPG_MIPS
                + "_"
                + "SCALING_"
                + data_type
                + "_"
                + "atoms"
                + str(num_atoms)
                + ".csv",
            )
            os.makedirs(log_dir, exist_ok=True)
            log_df.to_csv(filename, index=False)

    print(f"Mean accuracy of NAPG for {data_type}: {np.mean(avg_accuracy)} \n")
    return np.mean(avg_accuracy)