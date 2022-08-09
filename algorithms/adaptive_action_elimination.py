import numpy as np
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
from utils.utils import subset_2d, fit_and_plot
from utils.bandit_utils import get_ci


def adaptive_action_elimination(
    atoms: np.ndarray,
    signal: np.ndarray,
    var_proxy: np.ndarray,
    epsilon: float,
    delta: float,
    num_best_atoms: int = 1,
    verbose: bool = False,
    batch_size: int = BATCH_SIZE,
) -> Tuple[np.ndarray, int]:
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
    D = atoms.shape[0]  # Number of atoms
    d = atoms.shape[1]  # Dimension of atoms/signal
    t = 0  # time step
    new_delta = delta
    budgets = 0  # Number of calculations
    candidates = np.arange(D)  # arms (atoms)
    lcbs = -np.inf * np.ones(D)  # lower confidence bounds of arm returns
    ucbs = np.inf * np.ones(D)  # upper confidence bounds of arm returns
    means = np.zeros(D)  # estimated means of arm returns
    sorted_idcs = np.argsort(
        -np.abs(signal)
    )  # Indices that sorts signal in the descending order
    # l2_atoms = np.zeros(D)  # l2 norms of atoms


    while len(candidates) > num_best_atoms:
        t += 1

        # Find a true inner product value of candidates if we sample as much as their length (length of atom/signal)
        if t * batch_size > d:
            means[candidates] = np.abs(atoms[candidates] @ signal)
            budgets += d * len(candidates)
            return candidates[np.argsort(-means[candidates])][:num_best_atoms], budgets

        batch_size = min(
            batch_size, d - (t - 1) * batch_size
        )  # Can't sample more than population size
        sample_idcs = sorted_idcs[(t - 1) * batch_size: t * batch_size]
        signal_sampled = signal[sample_idcs]
        candidates_sampled = subset_2d(atoms, rows=candidates, columns=sample_idcs)
        means[candidates] = (
            (t * batch_size * means[candidates]) + (candidates_sampled @ signal_sampled)
        ) / ((t + 1) * batch_size)

        # Make confidence interval tighter using the constraint that all the vectors have unit length.
        # l2_atoms[candidates] += np.sum(
        #     np.square(candidates_sampled), axis=1
        # )  # l2 norm of sampled atoms
        max_signal = np.abs(signal_sampled[0])  # maximum of signal sampled
        # multiplication_bounds = (
        #     2 * (1 - l2_atoms[candidates]) * np.abs(max_signal)
        # )  # The bound of the multiplication
        # between each element from signal and atoms
        # new_var_proxy = (
        #     multiplication_bounds ** 2 / 4
        # )  # See Hoeffding's lemma (lemma 18) in https://bit.ly/3RXawav
        new_var_proxy = var_proxy
        new_var_proxy = np.minimum(var_proxy, new_var_proxy)
        ci_vec = get_ci(
            delta=new_delta,
            var_proxy=new_var_proxy,
            ci_bound=HOEFFDING,
            num_samples=t * batch_size,
        )
        budgets += batch_size * len(candidates)
        if np.all(4 * ci_vec < epsilon):
            return candidates[np.argsort(-means[candidates])][:num_best_atoms], budgets
        ucbs[candidates] = means[candidates] + ci_vec
        lcbs[candidates] = means[candidates] - ci_vec
        new_cand_idcs = np.where((ucbs[candidates] >= lcbs[candidates].max()))[0]

        if (
            len(new_cand_idcs) < num_best_atoms
        ):  # Have to return candidates with length num_best_atoms
            return candidates[np.argsort(-means[candidates])][:num_best_atoms], budgets
        candidates = candidates[new_cand_idcs]
    return candidates, budgets
