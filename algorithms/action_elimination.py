import numpy as np
import numba as nb
from typing import Tuple

from utils.constants import (
    BATCH_SIZE,
    ADVERSARIAL_CUSTOM,
    NORMAL_CUSTOM,
    ACTION_ELIMINATION,
    HOEFFDING,
)
from utils.utils import subset_2d, subset_2d_cached, fit_and_plot, set_seed
from utils.bandit_utils import get_ci


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
    seed: int=0,
    use_cache: bool = True,
    permutation: list = None,
    cache: np.ndarray = None,
    max_cache_size: int = 100,
    cache_tracker: np.ndarray = None
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
        # TODO: Maybe need to change this to a > 0 and ucbs > lcbs.max below
        # TODO: Don't let old candidates come back?
        # TODO: Change this to > 2, since if there is only one candidate left it will be the leader
        while len(candidates) > num_best_atoms:
            t += 1
            start = (t - 1) * batch_size
            end = t * batch_size

            # Compute the true inner product of the candidates if we have more samples than dimension d.
            # (i.e. no need to approximate from this point forward)
            if end > d:
                sample_indices = permutation[start:]
                signal_sampled = signal[sample_indices]
                candidates_sampled = subset_2d(atoms, rows=candidates, columns=sample_indices)
                means[candidates] = (
                    (start * means[candidates])
                    + (candidates_sampled @ signal_sampled)
                ) / d
                budgets += (d - start) * len(candidates)

                if abs:
                    candidates = candidates[np.argsort(-np.abs(means[candidates]))][:num_best_atoms]
                else:
                    candidates = candidates[np.argsort(-means[candidates])][:num_best_atoms]

                # At this point, we've computed all possible inner products.
                # Exit the loop.
                break

            # At this point, we know that the reference points will never be in the cache since we're using a fixed
            # permutation. In that case, we sample naively.
            if start > max_cache_size:
                use_cache = False

            sample_indices = permutation[start: end]
            if use_cache:
                candidates_sampled, cache, cache_tracker, budget = subset_2d_cached(
                    source_array=atoms,
                    rows=candidates,
                    columns=sample_indices,
                    start=start,
                    cache=cache,
                    cache_tracker=cache_tracker
                )
                budgets += budget
            else:
                candidates_sampled = subset_2d(source_array=atoms, rows=candidates, columns=sample_indices)
                budgets += batch_size * len(candidates)

            signal_sampled = signal[sample_indices]
            means[candidates] = (
                (start * means[candidates])
                + (candidates_sampled @ signal_sampled)
            ) / (
                end
            )  # Update the running means of arm returns # Does this np.dot do implicit broadcasting?

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
