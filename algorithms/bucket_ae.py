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


@nb.njit
def bucket_ae(
    atoms: np.ndarray,
    signals: np.ndarray,
    var_proxy: np.ndarray,
    epsilon: float,
    delta: float,
    num_best_atoms: int = 1,
    verbose: bool = False,
    batch_size: int = BATCH_SIZE,
    with_replacement: bool = True,
    seed=0,
    num_buckets: int = 1,
    num_samples_bucketing: int = 100,
    z_score: float = 3,
    # Scipy package is normally used for calculating z-score, but it is not compatible with numba
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Put atoms into k buckets with descending order of their norm. Then, solve MIPS-problem sequentially. If the lcb of
    inner product between the best atom and query is greater than the upper bound of remaining buckets( = norm of atom
    * norm of query vector), returns the best atom.

    This is to reduce unnecessary computations for other buckets.
    """
    rng = np.random.default_rng(seed)
    N = atoms.shape[0]  # Number of atoms
    d = atoms.shape[1]  # Dimension of atoms
    k = num_buckets
    m = int(N / (k - 1))
    q = num_samples_bucketing
    assert q <= d, (
        "number of samples to estimate the norm of each atom should be less than or"
        "equal to the dimension of atoms"
    )

    sample_idcs = rng.integers(low=0, high=d, size=num_samples_bucketing)
    sampled_atoms = atoms[:, sample_idcs]
    candidates = np.empty((signals.shape[0], num_best_atoms), dtype=np.int64)
    atoms_means = np.mean(atoms, axis=1)
    atoms_ub = atoms_means + z_score * np.sqrt(
        np.var(sampled_atoms, axis=1) / (q - 1) * (1 - q / N)
    )  # see shorturl.at/tABCP for the formula
    sort_idcs = np.argsort(atoms_ub)[:-1]
    ub_buckets = np.empty(k)  # Upper bound of each bucket
    bucket_idcs = []
    for idx in range(k):
        ub_buckets[idx] = atoms_ub[sort_idcs[idx * m]]
        bucket_idcs.append(sort_idcs[idx * m: min((idx + 1) * m, N)])

    candidates_array = np.empty((signals.shape[0], num_best_atoms), dtype=np.int64)
    budgets_array = np.empty(signals.shape[0], dtype=np.int64)
    for signal_idx in range(signals.shape[0]):
        signal = signals[signal_idx]
        means = np.empty(0)
        lcbs = np.empty(0)
        candidates = np.empty(0)
        budgets = 0
        for idx in range(k):
            temp_candidates, temp_means, temp_lcbs, temp_budgets = bucket_ae_helper(
                atoms[bucket_idcs[idx]],
                signal,
                var_proxy,
                epsilon,
                delta,
                num_best_atoms,
                verbose,
                batch_size,
                with_replacement,
                seed,
            )
            best_k_idcs = np.argsort(np.concatenate((means, temp_means)))[:-1][:num_best_atoms]
            candidates = np.concatenate((candidates, temp_candidates))[best_k_idcs]
            lcbs = np.concatenate((lcbs, temp_lcbs))[best_k_idcs]
            budgets += temp_budgets
            if np.max(lcbs) > ub_buckets[idx + 1]:
                break
        candidates_array[signal_idx] = candidates
        budgets_array[signal_idx] = budgets

    return candidates_array, budgets_array







@nb.njit
def bucket_ae_helper(
    atoms: np.ndarray,
    signal: np.ndarray,
    var_proxy: np.ndarray,
    epsilon: float,
    delta: float,
    num_best_atoms: int = 1,
    verbose: bool = False,
    batch_size: int = BATCH_SIZE,
    with_replacement: bool = True,
    seed=0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    A best-arm identification algorithm that eliminates the arms whose upper bound is less than the lower bound of the
    arm with highest estimated reward in each round. See algorithm 1 in https://arxiv.org/pdf/2006.06856.pdf and
    AE algorithm in https://homes.cs.washington.edu/~jamieson/resources/bestArmSurvey.pdf#page=2

    :return: Returns the k(num_best_atoms) best arms with error epsilon and probability 1 - delta.
    """
    set_seed(seed)
    D = atoms.shape[0]  # Number of atoms
    d = atoms.shape[1]  # Dimension of atoms/signal
    t = 0  # time step
    new_delta = delta * batch_size / (d * D)
    budgets = 0  # Number of calculations
    candidates = np.arange(D)  # arms (atoms)
    lcbs = -np.inf * np.ones(D)  # lower confidence bounds of arm returns
    ucbs = np.inf * np.ones(D)  # upper confidence bounds of arm returns
    means = np.zeros(D)  # estimated means of arm returns
    if not with_replacement:
        shuffled_idcs = np.arange(d)
        np.random.shuffle(shuffled_idcs)
    else:
        shuffled_idcs = None
    # TODO: Add exact computation if total mnist too big
    # TODO: Maybe need to change this to a >0 and ucbs > lcbs.max below
    # TODO: Don't let old candidates come back?
    # TODO: Change this to > 2, since if there is only one candidate left it will be the leader
    while len(candidates) > num_best_atoms:
        t += 1
        # Find a true inner product value of candidates if we sample more than their dimension (length of
        # atom/signal)
        if t * batch_size > d:
            if with_replacement:
                means[candidates] = atoms[candidates] @ signal
                budgets += d * len(candidates)
            else:
                sample_idcs = shuffled_idcs[(t - 1) * batch_size :]
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
        budgets += batch_size * len(candidates)

        if with_replacement:
            sample_idcs = np.random.choice(np.arange(d), batch_size, replace=False)
        else:
            sample_idcs = shuffled_idcs[(t - 1) * batch_size : t * batch_size]
        signal_sampled = signal[sample_idcs]
        candidates_sampled = subset_2d(atoms, rows=candidates, columns=sample_idcs)
        means[candidates] = (
            (t * batch_size * means[candidates])
            + (candidates_sampled @ signal_sampled)
        ) / (
            (t + 1) * batch_size
        )  # Update the running means of arm returns # Does this np.dot do implicit broadcasting?
        if with_replacement:
            ci = get_ci(
                delta=new_delta,
                var_proxy=var_proxy,
                ci_bound=HOEFFDING,
                num_samples=t * batch_size,
            )
        else:
            ci = get_ci(
                delta=new_delta,
                var_proxy=var_proxy,
                ci_bound=HOEFFDING,
                num_samples=t * batch_size,
                pop_size=d,
                with_replacement=False,
            )
        if 4 * ci < epsilon:
            sort_idcs = np.argsort(-means[candidates])
            candidates = candidates[sort_idcs][:num_best_atoms]
            candidates_lcbs = lcbs[sort_idcs][:num_best_atoms]
            candidates_means = means[sort_idcs][:num_best_atoms]
            break
        ucbs[candidates] = means[candidates] + ci
        lcbs[candidates] = means[candidates] - ci
        new_cand_idcs = np.where((ucbs[candidates] >= lcbs[candidates].max()))[0]

        if (
            len(new_cand_idcs) < num_best_atoms
        ):  # Have to return candidates with length num_best_atoms
            sort_idcs = np.argsort(-means[candidates])
            candidates = candidates[sort_idcs][:num_best_atoms]
            candidates_lcbs = lcbs[sort_idcs][:num_best_atoms]
            candidates_means = means[sort_idcs][:num_best_atoms]
            break
        candidates = candidates[new_cand_idcs]
        candidates_lcbs = lcbs[new_cand_idcs]
        candidates_means = means[new_cand_idcs]
    return candidates, candidates_means, candidates_lcbs, budgets
