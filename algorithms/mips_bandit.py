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
from data.custom_data import generate_normal_data, generate_adversarial_data


def mips_bandit(
    atoms: np.ndarray,
    signal: np.ndarray,
    maxmin: Tuple,
    bandit_alg: str,
    epsilon: float,
    delta: float,
    num_best_atoms: int = 1,
    seed: int = 0,
    is_experiment: bool = False,
    verbose: bool = False,
    with_replacement: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Solve MIPS(Maximum Inner Product Search) problem using a multi-armed bandit algorithm with fixed confidence setting.
    MIPS problem: To find a best atom vector in data_atoms
     set that has maximum inner product with a signal vector.

    :param atoms: 2d array that contains atoms
     in rows. atoms
      are the vectors one of which will be selected
    as the vector with maximum inner product with a signal vector.
    :param signal: A signal vector.
    :param maxmin: A tuple with two elements which are the maximum and minimum of data_atoms
     array
    :param bandit_alg: A name of multi-armed bandit algorithm
    :param epsilon: A maximum additive error between the true best arm and the best arm identified by our
    bandit algorithm
    :param delta: A probability that the difference between the reward of the true best arm and the identified
    best arm is beyond epsilon
    :param num_best_atoms
    : Number of atoms
     that bandit algorithm identify as top/best arms.
    :param seed: A random seed
    :param is_experiment: Whether to do worst-case experiment
    :param verbose: Verbosity of function
    :param with_replacement: Whether to sample with replacement
    :return: Indices of best atom that have maximum inner product with a signal vector and the budget (total number of
    calculations)
    """
    assert len(atoms.shape) == 2, "Invalid size of data atoms" " matrix"
    assert len(signal.shape) == 1, "Invalid size of signal vector"

    if with_replacement:
        if bandit_alg is ACTION_ELIMINATION:
            return action_elimination(
                atoms=atoms,
                signal=signal,
                var_proxy=1 / 4 * (maxmin[0] - maxmin[1]) ** 2,
                epsilon=epsilon,
                delta=delta,
                num_best_atoms=num_best_atoms,
            )
        else:
            raise NotImplementedError(f"{bandit_alg} is not implemented.")
    else:
        if bandit_alg is MEDIAN_ELIMINATION:
            return median_elimination(
                atoms=atoms,
                signal=signal,
                epsilon=epsilon,
                delta=delta,
                num_best_atoms=num_best_atoms,
                is_experiment=is_experiment,
                seed=seed,
            )
        elif bandit_alg is ACTION_ELIMINATION:
            return adaptive_action_elimination(
                atoms=atoms,
                signal=signal,
                var_proxy=1 / 4 * (maxmin[0] - maxmin[1]) ** 2,
                epsilon=epsilon,
                delta=delta,
                num_best_atoms=num_best_atoms,
            )
        else:
            raise NotImplementedError(f"{bandit_alg} is not implemented.")