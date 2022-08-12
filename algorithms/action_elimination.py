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
    signals: np.ndarray,
    var_proxy: np.ndarray,
    epsilon: float,
    delta: float,
    num_best_atoms: int = 1,
    verbose: bool = False,
    batch_size: int = BATCH_SIZE,
    with_replacement: bool = True,
    seed=0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    A best-arm identification algorithm that eliminates the arms whose upper bound is less than the lower bound of the
    arm with highest estimated reward in each round. See algorithm 1 in https://arxiv.org/pdf/2006.06856.pdf and
    AE algorithm in https://homes.cs.washington.edu/~jamieson/resources/bestArmSurvey.pdf#page=2

    :return: Returns the k(num_best_atoms) best arms with error epsilon and probability 1 - delta.
    """
    rng = np.random.default_rng(seed)
    candidates_array = []
    budgets_array = []
    for signal in signals:
        D = atoms.shape[0]  # Number of atoms
        d = atoms.shape[1]  # Dimension of atoms/signal
        t = 0  # time step
        new_delta = delta / (d * D)
        budgets = 0  # Number of calculations
        candidates = np.arange(D)  # arms (atoms)
        lcbs = -np.inf * np.ones(D)  # lower confidence bounds of arm returns
        ucbs = np.inf * np.ones(D)  # upper confidence bounds of arm returns
        means = np.zeros(D)  # estimated means of arm returns
        if not with_replacement:
            shuffled_idcs = np.arange(d)
            rng.shuffle(shuffled_idcs)
        else:
            shuffled_idcs = None
        # TODO: Add exact computation if total mnist too big
        # TODO: Maybe need to change this to a >0 and ucbs > lcbs.max below
        # TODO: Don't let old candidates come back?
        # TODO: Change this to > 2, since if there is only one candidate left it will be the leader
        while len(candidates) > num_best_atoms:
            t += 1

            # Find a true inner product value of candidates if we sample more than their dimension (length of atom/signal)
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
                (t * batch_size * means[candidates]) + (candidates_sampled @ signal_sampled)
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
                candidates = candidates[np.argsort(-means[candidates])][:num_best_atoms]
                break
            ucbs[candidates] = means[candidates] + ci
            lcbs[candidates] = means[candidates] - ci
            new_cand_idcs = np.where((ucbs[candidates] >= lcbs[candidates].max()))[0]

            if (
                len(new_cand_idcs) < num_best_atoms
            ):  # Have to return candidates with length num_best_atoms
                candidates = candidates[np.argsort(-means[candidates])][:num_best_atoms]
                break
            candidates = candidates[new_cand_idcs]
        candidates_array.append(candidates)
        budgets_array.append(budgets)
    return np.array(candidates_array), np.array(budgets_array)
