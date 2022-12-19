import os
import pandas as pd
import time
from typing import Tuple, Optional
import numpy as np
from collections import defaultdict
from typing import Tuple
from utils.constants import (
    SCALING_NUM_ATOMS,
    SCALING_NUM_SIGNALS,
    SCALING_SIZE_MINMAX,
    NORMALIZED_MAXMIN,
    ACTION_ELIMINATION,
)

from algorithms.mips_bandit import mips_bandit

from utils.constants import (
    NORMAL_CUSTOM,
    MEDIAN_ELIMINATION,
    HNSW_MIPS,
    NAHNSW_MIPS,

    NUM_SEEDS,
    SCALING_NUM_EXPERIMENTS,
    DEFAULT_MAXMIN,

    SCALING_DELTA,
    SCALING_TOPK,
    SCALING_EPSILON,
)

from utils.utils import get_sizes_arr
from data.get_data import get_data
import hnswlib

def scaling_exp(
    data_type: str = NORMAL_CUSTOM,
    mips_alg: str = MEDIAN_ELIMINATION,
    num_experiments: int = SCALING_NUM_EXPERIMENTS,
    num_seeds: int = NUM_SEEDS,
    data_generation_seed: int = 0,
    size_minmax: Tuple[int, int] = (10, 10**6),
    num_atoms: int = 10**3,
    num_signals: int = 1,
    num_best_atoms: int = SCALING_TOPK,
    is_logspace: bool = False,
) -> None:
    """
    Run the time complexity of the bandit and baseline algorithms as the signal size scales and store them in
    the appropriate log files. MEDIAN_ELIMINATION algorithm should satisfy this scaling according to Corollary 3 in
    "https://arxiv.org/pdf/1812.06360.pdf#page=5".

    :param epsilon: A epsilon (an additive error allowed)
    :param delta: A delta (an upper bound of failure probability)
    :param maxmin: Maximum and minimum of data
    :param data_type: A name of data generating function
    :param mips_alg: A name of algorithm
    :param num_experiments: Number of experiments
    :param num_seeds: Number of seeds
    :param data_generation_seed: Random seed for generating data
    :param size_minmax: Minimum and Maximum of signal vector size
    :param num_atoms: Number of atoms
    :param num_signals: Number of signals
    :param num_best_atoms: Number of atoms we rank in the descending order of inner product
    :param with_replacement: Whether to sample with replacement
    :param is_log: Whether to log the results
    :param is_plot: Whether to plot the results
    :param is_logspace: Whether to construct sizes of signal spaced evenly on a log scale
    :param dirname: Directory name to log the results
    """
    global budget_array

    # average over NUM_SEEDS results
    accuracy_list_per_dimension = defaultdict(list)
    print(f"=> Scaling experiment for {mips_alg} with {data_type} dataset")

    sizes_signal = get_sizes_arr(is_logspace, size_minmax, num_experiments=num_experiments)
    atoms, signals = get_data(
        num_atoms=num_atoms,
        len_signal=sizes_signal[-1],
        num_signals=num_signals,
        seed=data_generation_seed,
        data_type=data_type,
    )
    for seed in range(num_seeds):
        print(f"... running seed {seed}")
        rng = np.random.default_rng(seed)

        # run the experiment for each signal size
        for size in sizes_signal:
            subset_indices = rng.choice(sizes_signal[-1], size, replace=False)
            atoms_subset, signals_subset = (
                atoms[:, subset_indices],
                signals[:, subset_indices],
            )

            index = hnswlib.Index(space="ip", dim=size, useNormFactor_=(mips_alg==NAHNSW_MIPS))

            # Initializing index - the maximum number of elements should be known beforehand
            index.init_index(max_elements=num_atoms, ef_construction=200, M=16)

            # Element insertion (can be called several times):
            ids = np.arange(num_atoms)
            index.add_items(atoms_subset, ids)

            labels, distances, budget = index.knn_query(signals_subset, k=num_best_atoms)
            budget_array = np.array(budget)
            candidates_array = np.sort(np.array(labels))

            # store runtime, budget, budget_std, and accuracy of the experiment
            naive_candidates_array = np.sort(
                np.matmul(-atoms_subset, signals_subset.transpose())
                .argsort(axis=0)[:num_best_atoms]
                .transpose()
            )

            accuracy = np.mean(naive_candidates_array==candidates_array)
            accuracy_list_per_dimension[size].append(accuracy)

    for (dimension, accuracy_list) in accuracy_list_per_dimension.items():
        print(f'{mips_alg} ::: Dimension {dimension} => accuracy: {np.mean(accuracy_list)}')


if __name__ == "__main__":
    scaling_exp(
        num_atoms=SCALING_NUM_ATOMS,
        size_minmax=SCALING_SIZE_MINMAX,
        num_experiments=SCALING_NUM_EXPERIMENTS,
        num_signals=SCALING_NUM_SIGNALS,
        data_type=NORMAL_CUSTOM,
        mips_alg=HNSW_MIPS,
        is_logspace=True,
    )

    scaling_exp(
        num_atoms=SCALING_NUM_ATOMS,
        size_minmax=SCALING_SIZE_MINMAX,
        num_experiments=SCALING_NUM_EXPERIMENTS,
        num_signals=SCALING_NUM_SIGNALS,
        data_type=NORMAL_CUSTOM,
        mips_alg=NAHNSW_MIPS,
        is_logspace=True,
    )