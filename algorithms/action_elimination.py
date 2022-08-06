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


def action_elimination(
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
    A best-arm identification algorithm that eliminates the arms whose upper bound is less than the lower bound of the
    arm with highest estimated reward in each round. See algorithm 1 in https://arxiv.org/pdf/2006.06856.pdf and
    AE algorithm in https://homes.cs.washington.edu/~jamieson/resources/bestArmSurvey.pdf#page=2

    :return: Returns the k(num_best_atoms) best arms with error epsilon and probability 1 - delta.
    """
    D = atoms.shape[0]  # Number of atoms
    d = atoms.shape[1]  # Dimension of atoms/signal
    t = 0  # time step
    new_delta = delta / (d * D)
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
            means[candidates] = np.abs(atoms[candidates] @ signal)
            budgets += d * len(candidates)
            return candidates[np.argsort(-means[candidates])][:num_best_atoms], budgets
        budgets += batch_size * len(candidates)

        sample_idcs = np.random.choice(np.arange(d), batch_size, replace=False)
        signal_sampled = signal[sample_idcs]
        candidates_sampled = subset_2d(atoms, rows=candidates, columns=sample_idcs)
        means[candidates] = (
            (t * batch_size * means[candidates]) + (candidates_sampled @ signal_sampled)
        ) / (
            (t + 1) * batch_size
        )  # Update the running means of arm returns # Does this np.dot do implicit broadcasting?
        ci = get_ci(
            delta=new_delta,
            var_proxy=var_proxy,
            ci_bound=HOEFFDING,
            num_samples=t * batch_size,
        )
        if 4 * ci < epsilon:
            return candidates[np.argsort(-means[candidates])][:num_best_atoms], budgets
        ucbs[candidates] = means[candidates] + ci
        lcbs[candidates] = means[candidates] - ci
        new_cand_idcs = np.where((ucbs[candidates] >= lcbs[candidates].max()))[0]

        if (
            len(new_cand_idcs) < num_best_atoms
        ):  # Have to return candidates with length num_best_atoms
            return candidates[np.argsort(-means[candidates])][:num_best_atoms], budgets
        candidates = candidates[new_cand_idcs]
    return candidates, budgets
