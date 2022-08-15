import os
import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from utils.constants import (
    BATCH_SIZE,
    ADVERSARIAL_CUSTOM,
    NORMAL_CUSTOM,
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    HOEFFDING,
    GREEDY_MIPS,
)
from utils.utils import subset_2d, fit_and_plot
from utils.bandit_utils import get_ci
from data.get_data import get_data

from algorithms.mips_bandit import mips_bandit
from algorithms.greedy_mips import greedy_mips


def speedup_precision_exps(
    data_type: str = ADVERSARIAL_CUSTOM,
    mips_alg: str = MEDIAN_ELIMINATION,
    len_signals_list: np.ndarray = np.array([10 ** 5]),
    num_atoms_list: np.ndarray = np.array([10 ** 3]),
    num_experiments: int = 5,
    num_signals: int = 1,
    num_best_atoms: int = 1,
    seed: int = 0,
    with_replacement: bool = False,
    is_log: bool = True,
    is_plot: bool = True,
    dirname: str = None,
):
    rng = np.random.default_rng(seed)
    atoms, signals = get_data(
        num_atoms=num_atoms_list[-1],
        len_signal=len_signals_list[-1],
        num_signals=num_signals,
        seed=seed,
        data_type=data_type,
    )
    maxmin = (5, 0)
    for num_atoms in num_atoms_list:
        for len_signal in len_signals_list:
            speedup_list = []
            accuracy_list = []
            num_atoms_subset_idcs = rng.choice(
                num_atoms_list[-1], num_atoms, replace=False
            )
            len_signals_subset_idcs = rng.choice(
                len_signals_list[-1], len_signal, replace=False
            )
            atoms_subset, signals_subset = (
                subset_2d(atoms, num_atoms_subset_idcs, len_signals_subset_idcs),
                subset_2d(signals, np.arange(num_signals), len_signals_subset_idcs),
            )
            naive_runtime = 0
            for _ in range(num_experiments):
                start_time = time.time()
                naive_candidates_array = (
                    np.matmul(-atoms_subset, signals_subset.transpose())
                    .argsort(axis=0)[:num_best_atoms]
                    .transpose()
                )
                naive_runtime += time.time() - start_time
            naive_runtime /= num_experiments
            for epsilon in (0.00001, 0.01, 0.1):
                for delta in (0.01, 0.05, 0.3, 0.5):
                    our_average_runtime = 0
                    our_average_accuracy = 0
                    for seed in range(num_experiments):
                        if mips_alg in (ADAPTIVE_ACTION_ELIMINATION, ACTION_ELIMINATION, MEDIAN_ELIMINATION):
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
                            our_runtime = time.time() - start_time
                        elif mips_alg is GREEDY_MIPS:
                            start_time = time.time()
                            candidates_array, _, _, budget_array = greedy_mips(
                                atoms=atoms_subset,
                                signals=signals_subset,
                                budget=50,
                                num_best_atoms=num_best_atoms,
                            )
                            our_runtime = time.time() - start_time
                        else:
                            raise NotImplementedError(f"{mips_alg} is not implemented")
                        our_average_runtime += our_runtime
                        accuracy = 0
                        for idx in range(num_signals):
                            accuracy += len(
                                np.intersect1d(
                                    naive_candidates_array[idx], candidates_array[idx]
                                )
                            )
                        accuracy /= num_signals * num_best_atoms
                        our_average_accuracy += accuracy
                    our_average_runtime /= num_experiments
                    our_average_accuracy /= num_experiments
                    accuracy_list.append(our_average_accuracy)
                    speedup_list.append(naive_runtime / our_average_runtime)

            accuracy_list = np.array(accuracy_list)
            speedup_list = np.array(speedup_list)

            if is_log:
                log_dict = {
                    "data_type": data_type,
                    "num_atoms": num_atoms,
                    "len_signal": len_signal,
                    "speedup": speedup_list,
                    "accuracy": accuracy_list,
                }
                log_df = pd.DataFrame(log_dict)
                log_dir = "logs" if dirname is None else dirname
                filename = os.path.join(
                    log_dir,
                    "SPEEDUP_"
                    + data_type
                    + "_"
                    + mips_alg
                    + "_"
                    + str((num_atoms, len_signal)),
                )
                os.makedirs(log_dir, exist_ok=True)
                log_df.to_csv(filename, index=False)
            if is_plot:
                idcs = np.argsort(speedup_list)
                plt.plot(speedup_list[idcs], accuracy_list[idcs], marker='o')
                plt.title(f"{mips_alg}, data size: {(num_atoms, len_signal)}")
                plt.ylabel("Accuracy")
                plt.xlabel(f"Speedup over naive approaches{naive_runtime}(s)")
                plt.show()
