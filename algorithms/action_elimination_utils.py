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


def compute_sampling_weights(
        signals: np.ndarray,
        atoms: np.ndarray,
        N_subsampled_atoms: int,
        alpha: float = 1,
        type: str = "ATOM",
) -> np.ndarray:
    """
    Compute the sampling weights for each coordinate
    """
    n = len(atoms)
    atom_idcs = np.random.choice(n, N_subsampled_atoms, replace=False)
    atoms_subsampled = atoms[atom_idcs]
    v2 = atoms_subsampled ** 2
    E_v2 = np.mean(v2, axis=0)  # Average across atoms

    # TODO: Is this the correct thing to do in the case of multiple signals?
    q2 = signals ** 2
    E_q2 = np.mean(q2, axis=0)  # Average across signals

    if alpha == 0.0:
        w_j = E_q2
    elif alpha == 0.5:
        w_j = np.sqrt(E_q2 * E_v2)
    elif alpha == 1:
        w_j = E_v2
    else:
        raise Exception("Unsupported alpha")

    w_j /= np.sum(w_j)
    return w_j


# TODO: May not want this njit here since we only call once as we're doing all signals at once?
#@nb.njit
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

    # TODO: Update budget computations throughout

    num_signals = len(signals)
    N = len(atoms)
    d = signals.shape[1]  # Need shape[1] here because there are multiple signals

    c = warm_start_num_idcs
    tmp_batch_size = min(d, c)
    warm_start_idcs = np.random.choice(d, tmp_batch_size, replace=False)
    budget = N * tmp_batch_size / num_signals # per-signal budget
    budget = 0

    # Do the large matrix multiplication

    # Don't need to call subset_2D here because we're doing all signals and atoms at once
    sampled_signals = signals[:, warm_start_idcs]
    sampled_atoms = atoms[:, warm_start_idcs]

    # means is going to be means[signal_idx][atom_idx]
    # We need the atoms' transpose so that we can do the dot product / matrix multiplication properly
    # This way, each row of sampled_signals is contracted with a row (column) of sampled_atoms (sampled_atoms.T)
    # The elements of the result are now mean_ij = <signal_i, atom_j>
    means_all = sampled_signals.dot(sampled_atoms.T)
    means_all /= d  # Need to divide by d to get per-coordinate mean
    assert means_all.shape == (num_signals, N)

    # Now need to compute lcbs and ucbs:
    # Get the variance proxies of the atoms

    # For each signal, multiply by the variance proxies by max(atom)**2
    # Compute CIs and filter candidates
    # lcbs, ucbs are of shape (num_signals, num_atoms)

    # Note: we're using mean here, which requires an O(num_signals*d) cost
    # TODO: Make this max or min?
    mean_signals = np.max(sampled_signals, axis=1)
    sum_var_proxy_all = np.outer(mean_signals ** 2, var_proxy)  # Index by signal_idx, atom_idx
    candidates_all = {}
    ci_vec_all = dict()
    lcbs_all = np.empty((num_signals, N))
    ucbs_all = np.empty((num_signals, N))
    for signal_idx, signal in enumerate(signals):
        # TODO: Can we change this to array broadcasting? At least it's compiled with numba
        ci_vec_all[signal_idx] = get_ci(
            delta=new_delta,
            num_samples=tmp_batch_size,
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
