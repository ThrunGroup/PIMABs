import numpy as np
import numba as nb
import math
from typing import Tuple

from numba.typed import Dict
from numba import types


from utils.constants import (
    BATCH_SIZE,
    ADVERSARIAL_CUSTOM,
    NORMAL_CUSTOM,
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    HOEFFDING,
)
from utils.utils import subset_2d, fit_and_plot, set_seed
from utils.bandit_utils import get_ci

from algorithms.action_elimination_utils import (
    compute_sampling_weights,
    warm_start
)

#@nb.njit
def adaptive_action_elimination(
    atoms: np.ndarray,
    signals: np.ndarray,
    var_proxy: np.ndarray,
    epsilon: float,
    delta: float,
    num_best_atoms: int = 1,
    verbose: bool = False,
    batch_size: int = BATCH_SIZE,
    seed: int = 0,
    do_warm_start: bool = True,
    warm_start_num_idcs: int = BATCH_SIZE, # We need to have warm_start_num_indices passed with a default int value because a None makes numba complain at compile time
) -> Tuple[np.ndarray, np.ndarray]:
    """
    An adaptive AE(Action Elimination) algorithm that adopts sampling without replacement, adaptive variance proxy, and
    smart sampling on top of original AE algorithm. It applies to MIPS(Maximum Inner product search) problem adaptively.

    Adaptive changes
    -Sample without replacement: Get a tight confidence interval by sampling without replacement, see lemma 1 in
    https://arxiv.org/pdf/1309.4029.pdf
    -Adaptive variance proxy: Tracking the sampled elements of signal vector and atoms, we can make the bounds of
    arms' rewards(dot product of atom and signal whose elements are sampled) tighter.
    -Smart sampling: By sorting the signal array, we can make confidence interval tighter and can expect the faster
    best arm identification.

    Should assert that all vectors are normalized. Omit this assertion statement in function due to its hevay
    computation.

    :return: Returns the k(num_best_atoms) best arms with error epsilon and probability 1 - delta.
    """
    set_seed(seed)
    candidates_array = np.empty((signals.shape[0], num_best_atoms), dtype=np.int64)
    budgets_array = np.empty(signals.shape[0], dtype=np.int64)

    D = atoms.shape[0]  # Number of atoms
    d = atoms.shape[1]  # Dimension of atoms/signal
    new_delta = delta * batch_size / (d * D)

    if do_warm_start:
        budget_all, candidates_all, means_all, lcbs_all, ucbs_all, sum_var_proxy_all = warm_start(signals, atoms, var_proxy, warm_start_num_idcs, new_delta)

    for signal_idx, signal in enumerate(signals):
        t = 0  # time step
        if do_warm_start:
            candidates = candidates_all[signal_idx]
            budgets = budget_all  # It's already an int, per-signal budget
            means = means_all[signal_idx]
            lcbs = lcbs_all[signal_idx]
            ucbs = ucbs_all[signal_idx]
            sum_var_proxy = sum_var_proxy_all[signal_idx]
        else:
            budgets = 0  # Number of calculations
            candidates = np.arange(D)  # arms (atoms)
            means = np.zeros(D)  # estimated means of arm returns
            lcbs = -np.inf * np.ones(D)  # lower confidence bounds of arm returns
            ucbs = np.inf * np.ones(D)  # upper confidence bounds of arm returns
            sum_var_proxy = 0

        # Get the indices of the signal in descending order (- sign -> descending order)
        # Do this after warm start because the sorting order will depend on the signal, can't do it across all
        # signals simultaneously

        # TODO: Add dlogd to budget here
        sorted_idcs = np.argsort(-np.abs(signal))
        while len(candidates) > num_best_atoms:
            t += 1
            # Find a true inner product value of candidates if we sample as much as their length (length of atom/signal)
            if t * batch_size >= d:
                sample_idcs = sorted_idcs[(t - 1) * batch_size :]
                signal_sampled = signal[sample_idcs]
                candidates_sampled = subset_2d(
                    atoms, rows=candidates, columns=sample_idcs
                )
                means[candidates] = (
                    (t * batch_size * means[candidates])
                    + (candidates_sampled @ signal_sampled)
                ) / d
                budgets += (d - (t - 1) * batch_size) * len(candidates)
                candidates = candidates[np.argsort(-means[candidates])][:num_best_atoms]
                break

            sample_idcs = sorted_idcs[(t - 1) * batch_size : t * batch_size]
            signal_sampled = signal[sample_idcs]
            candidates_sampled = subset_2d(atoms, rows=candidates, columns=sample_idcs)
            means[candidates] = (
                (t * batch_size * means[candidates])
                + (candidates_sampled @ signal_sampled)
            ) / ((t + 1) * batch_size)

            # First coordinate will be the largest
            max_signal = np.abs(signal_sampled[0])
            sum_var_proxy += var_proxy * (max_signal**2)
            ci_vec = get_ci(
                delta=new_delta,
                num_samples=(t * batch_size),
                ci_bound=HOEFFDING,
                pop_size=signal.shape[0],
                with_replacement=False,
                sum_var_proxy=sum_var_proxy,
            )
            budgets += batch_size * len(candidates)
            if 4 * ci_vec < epsilon:
                candidates = candidates[np.argsort(-means[candidates])][:num_best_atoms]
                break

            ucbs[candidates] = means[candidates] + ci_vec
            lcbs[candidates] = means[candidates] - ci_vec
            new_cand_idcs = np.where((ucbs[candidates] >= lcbs[candidates].max()))[0]
            # Have to return candidates with length num_best_atoms
            if (len(new_cand_idcs) < num_best_atoms):
                candidates = candidates[np.argsort(-means[candidates])][:num_best_atoms]
                break

            candidates = candidates[new_cand_idcs]

        candidates_array[signal_idx] = candidates
        budgets_array[signal_idx] = budgets

    return candidates_array, budgets_array
