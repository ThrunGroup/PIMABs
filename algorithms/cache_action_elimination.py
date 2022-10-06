import numpy as np
import numba as nb
import math
from collections import defaultdict
from typing import Tuple

from utils.constants import (
    BATCH_SIZE,
    ADVERSARIAL_CUSTOM,
    NORMAL_CUSTOM,
    ACTION_ELIMINATION,
    HOEFFDING,
)
from utils.utils import subset_2d, fit_and_plot, set_seed
from utils.bandit_utils import get_ci


"""

TODO:
1. update budget computation
2. solve "TypeError: some keyword arguments unexpected" error

"""


@nb.njit
def action_elimination(
    atoms: np.ndarray,
    signals: np.ndarray,
    var_proxy: np.ndarray,
    epsilon: float,
    delta: float,
    abs: bool,
    num_best_atoms: int = 1,
    verbose: bool = False,
    batch_size: int = BATCH_SIZE,
    seed=0,
    use_cache: bool = True,
    shuffled_indices: list = None,
    cache: defaultdict = None,
    cache_multiplier: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    A best-arm identification algorithm that eliminates the arms whose upper bound is less than the lower bound of the
    arm with the highest estimated reward in each round. See algorithm 1 in https://arxiv.org/pdf/2006.06856.pdf and
    AE algorithm in https://homes.cs.washington.edu/~jamieson/resources/bestArmSurvey.pdf#page=2

    :return: Returns the k(num_best_atoms) best arms with error epsilon and probability 1 - delta.
    """
    set_seed(seed)
    candidates_array = np.empty((signals.shape[0], num_best_atoms), dtype=np.int64)
    budgets_array = np.empty(signals.shape[0], dtype=np.int64)
    for signal_idx in range(signals.shape[0]):
        signal = signals[signal_idx]
        D = atoms.shape[0]  # Number of atoms
        d = atoms.shape[1]  # Dimension of atoms/signal
        t = 0  # time step
        new_delta = delta * batch_size / (d * D)
        budgets = 0  # Number of calculations
        candidates = np.arange(D)  # arms (atoms)
        lcbs = -np.inf * np.ones(D)  # lower confidence bounds of arm returns
        ucbs = np.inf * np.ones(D)  # upper confidence bounds of arm returns
        means = np.zeros(D)  # estimated means of arm returns

        # TODO: Add exact computation if total mnist too big
        # TODO: Maybe need to change this to a >0 and ucbs > lcbs.max below
        # TODO: Don't let old candidates come back?
        # TODO: Change this to > 2, since if there is only one candidate left it will be the leader
        while len(candidates) > num_best_atoms:
            t += 1
            # Find a true inner product value of candidates if we sample more than their dimension (length of atom/signal)
            if t * batch_size > d:
                sample_indices = shuffled_indices[(t - 1) * batch_size :]
                signal_sampled = signal[sample_indices]
                candidates_sampled = subset_2d(
                    atoms, rows=candidates, columns=sample_indices
                )
                means[candidates] = (
                    (t * batch_size * means[candidates])
                    + (candidates_sampled @ signal_sampled)
                ) / d
                budgets += (d - (t - 1) * batch_size) * len(candidates)

                if abs:
                    candidates = candidates[np.argsort(-np.abs(means[candidates]))][:num_best_atoms]
                else:
                    candidates = candidates[np.argsort(-means[candidates])][:num_best_atoms]
                break
            budgets += batch_size * len(candidates)

            start_index = (t - 1) * batch_size
            end_index = t * batch_size
            sample_indices = shuffled_indices[start_index: end_index]

            if use_cache and end_index <= cache_multiplier * np.log10(D):
                # TODO: worst case get batch_size disadvantage
                # read cache
                candidates_sampled = np.empty((len(candidates), batch_size))
                for (i, candidate) in enumerate(candidates):
                    if len(cache[candidate]) < end_index:
                        # write cache
                        cache[candidate] += atoms[candidate][sample_indices],
                    candidates_sampled[i] = cache[candidate][start_index:end_index]
            else:
                candidates_sampled = subset_2d(atoms, rows=candidates, columns=sample_indices)

            signal_sampled = signal[sample_indices]
            means[candidates] = (
                (t * batch_size * means[candidates])
                + (candidates_sampled @ signal_sampled)
            ) / (
                (t + 1) * batch_size
            )  # Update the running means of arm returns
               # Does this np.dot do implicit broadcasting?

            ci = get_ci(
                delta=new_delta,
                var_proxy=var_proxy,
                ci_bound=HOEFFDING,
                num_samples=t * batch_size,
                pop_size=d,
                with_replacement=False,
            )

            if 4 * ci < epsilon:
                if abs:
                    candidates = candidates[np.argsort(-np.abs(means[candidates]))][:num_best_atoms]
                else:
                    candidates = candidates[np.argsort(-means[candidates])][:num_best_atoms]
                break
            if abs:
                ucbs[candidates] = np.abs(means[candidates]) + ci
                lcbs[candidates] = np.abs(means[candidates]) - ci
                new_cand_idcs = np.where((ucbs[candidates] >= lcbs[candidates].max()))[0]
            else:
                ucbs[candidates] = means[candidates] + ci
                lcbs[candidates] = means[candidates] - ci
                new_cand_idcs = np.where((ucbs[candidates] >= lcbs[candidates].max()))[0]

            if (
                len(new_cand_idcs) < num_best_atoms
            ):  # Have to return candidates with length num_best_atoms
                if abs:
                    candidates = candidates[np.argsort(-np.abs(means[candidates]))][:num_best_atoms]
                else:
                    candidates = candidates[np.argsort(-means[candidates])][:num_best_atoms]
                break
            candidates = candidates[new_cand_idcs]
        candidates_array[signal_idx] = candidates
        budgets_array[signal_idx] = budgets
    return candidates_array, budgets_array
