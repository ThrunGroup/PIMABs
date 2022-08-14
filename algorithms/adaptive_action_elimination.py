import numpy as np
import numba as nb
import math
from typing import Tuple

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

### Taken from https://github.com/numba/numba/issues/1269#issuecomment-472574352
@nb.njit
def np_apply_along_axis(func1d, axis, arr):
  assert arr.ndim == 2
  assert axis in [0, 1]
  if axis == 0:
    result = np.empty(arr.shape[1])
    for i in range(len(result)):
      result[i] = func1d(arr[:, i])
  else:
    result = np.empty(arr.shape[0])
    for i in range(len(result)):
      result[i] = func1d(arr[i, :])
  return result

@nb.njit
def np_mean(array, axis):
  return np_apply_along_axis(np.mean, axis, array)

@nb.njit
def np_std(array, axis):
  return np_apply_along_axis(np.std, axis, array)



# TODO: May not want this njit here since we only call once as we're doing all signals at once?
@nb.njit
def warm_start(
        signals: np.ndarray,
        atoms: np.ndarray,
        var_proxy: np.ndarray,
        warm_start_num_idcs: int = BATCH_SIZE,
        new_delta: float = None,
):
    """
    Warm start the algorithm by doing a large matrix multiplication for all signals and atoms, for a set of dimensions.
    This is a form of preprocessing. Note that it incurs num_signals*N*c cost. We have to decide how to include this in our
    measurements. According to what we've written in the paper, we can make this N*c cost because we cache the atom
    values? So the true complexity measures are the number of ATOM values we look at (not number of signal values).
    This is justifiable if N >> 1 and N >> num_signals.

    :param signals: Signal vector to warm start for
    :param atoms: Atoms with which to compute inner product
    :param var_proxy: Variance proxy for each ATOM only (without considering signal)
    :param warm_start_idcs: randomly sampled indices to warm start for
    :return: warm start candidates, budgets, lcbs, ucbs, and means
    """
    num_signals = len(signals)
    N = len(atoms)
    d = signals.shape[1]  # Need shape[1] here because there are multiple signals

    c = warm_start_num_idcs
    warm_start_idcs = np.random.choice(d, warm_start_num_idcs, replace=False)
    budget = N * c  # per-signal budget

    # Do the large matrix multiplication

    # Don't need to call subset_2D here because we're doing all signals and atoms at once
    sampled_signals = signals[:, warm_start_idcs]
    sampled_atoms = atoms[:, warm_start_idcs]

    # means is going to be means[signal_idx][atom_idx]
    # We need the atoms' transpose so that we can do the dot product / matrix multiplication properly
    # This way, each row of sampled_signals is contracted with a row (column) of sampled_atoms (sampled_atoms.T)
    # The elements of the result are now mean_ij = <signal_i, atom_j>
    means_all = sampled_signals.dot(sampled_atoms.T)
    assert means_all.shape == (num_signals, N)

    # Now need to compute lcbs and ucbs:
    # Get the variance proxies of the atoms

    # For each signal, multiply by the variance proxies by max(atom)**2
    # Compute CIs and filter candidates
    # lcbs, ucbs are of shape (num_signals, num_atoms)

    # Note: we're using mean here, which requires an O(num_signals*d) cost
    # TODO: Make this max or min?
    mean_signals = np_mean(sampled_signals, axis=1)
    sum_var_proxy_all = np.outer(mean_signals ** 2, var_proxy)  # Index by signal_idx, atom_idx
    candidates_all = {}
    ci_vec_all = {}
    lcbs_all = np.empty((num_signals, N))
    ucbs_all = np.empty((num_signals, N))
    for signal_idx, signal in enumerate(signals):
        # TODO: Can we change this to array broadcasting? At least it's compiled with numba
        ci_vec_all[signal_idx] = get_ci(
            delta=new_delta,
            num_samples=c,
            ci_bound=HOEFFDING,
            with_replacement=False,
            pop_size=d,
            sum_var_proxy=sum_var_proxy_all[signal_idx],
        )
        assert ci_vec_all[signal_idx].shape == (1,) # We have a scalar value since we use a single variance proxy for each

        # Implicit broadcasting: lcbs_all[signal_idx] and means_all[signal_idx] are of shape (N, 1) but ci_vec_all[signal_idx] is of shape (1,).
        lcbs_all[signal_idx] = means_all[signal_idx] - ci_vec_all[signal_idx]
        ucbs_all[signal_idx] = means_all[signal_idx] + ci_vec_all[signal_idx]
        candidates_all[signal_idx] = np.where((ucbs_all[signal_idx] >= lcbs_all[signal_idx].max()))[0]

    return budget, candidates_all, means_all, lcbs_all, ucbs_all, sum_var_proxy_all

@nb.njit
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
    do_warm_start: bool = False,
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
        signal = signals[signal_idx]
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
