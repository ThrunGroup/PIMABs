import numpy as np
import numpy.linalg as la
from typing import Tuple
from sklearn.linear_model import OrthogonalMatchingPursuit  # for debugging purposes

from algorithms.mips_bandit import mips_bandit
from data.get_data import get_data
from utils.utils import set_seed
from utils.constants import (
    NORMAL_CUSTOM,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
)


def matching_pursuit(
    atoms: np.ndarray,
    signal: np.ndarray,
    num_decomp: int = 5,
    min_residual: float = 0.001,
    is_omp: bool = False,
    bandit_alg: str = "",
    seed: int = 0,
) -> Tuple[np.ndarray, int]:
    """
    Runs the Matching Pursuit algorithm. Refer to implementation by https://github.com/stonemason11
    :param atoms: Atoms array
    :param signal: Signal vector
    :param num_decomp: number of decompositions we want to make (equivalent to scikit-learn's n_nonzero_coefs)
    :param min_residual: the threshold for which to terminate
    :param is_omp: Boolean flag that determines whether to run orthogonal matching pursuit
    :param bandit_alg: Name of the bandit algorithm to get MIPS. Default naive.
    :param seed: Seed

    :return: an array of coefficients similar to scikit-learn OMP's .coef_ attribute AND budget
    """
    set_seed(seed)
    decomp = 0
    total_budget = 0
    unexplored = np.arange(atoms.shape[0])
    candidates = np.zeros(num_decomp, dtype=np.int64)
    residual = signal

    while decomp < num_decomp and la.norm(residual) > min_residual:
        # solve the MIPS problem
        if len(bandit_alg) > 0:
            candidate, budget = mips_bandit(
                atoms[unexplored],  # exclude previous decompositions
                np.expand_dims(residual, axis=0),  # need to pass in 2d array
                maxmin=[-1, 1],
                bandit_alg=bandit_alg,
                epsilon=0.01,
                delta=0.01,
            )
        else:
            #import ipdb; ipdb.set_trace()
            candidate = np.argmax(atoms[unexplored] @ residual)
            budget = atoms[unexplored].shape[0] * atoms[unexplored].shape[1]

        # update candidates and exclude them from atoms
        candidates[decomp] = candidate
        unexplored = np.delete(unexplored, candidate, axis=0)
        total_budget += budget
        decomp += 1

        # update the residual
        if is_omp:
            mips_candidates = candidates[:decomp]   # candidates found so far
            #import ipdb; ipdb.set_trace()
            selected_atoms = atoms[mips_candidates]
            pseudo_inv = la.inv(selected_atoms.T @ selected_atoms) @ selected_atoms.T
            projection = pseudo_inv.T @ signal

            assert projection.shape[0] == selected_atoms.shape[0]
            residual = signal - selected_atoms.T @ projection

        else:
            residual -= (residual @ atoms[candidate]) * atoms[candidate]

    return candidates, total_budget


if __name__ == "__main__":
    atoms, signals = get_data(
        num_atoms=100,
        len_signal=1000,
        seed=0,
        data_type=NORMAL_CUSTOM,
        num_signals=1,
    )
    # scikit-learn
    model = OrthogonalMatchingPursuit(normalize=False, n_nonzero_coefs=5)
    print("------- running scikit-learn OMP ---------")
    model.fit(atoms.T, signals[0])
    print("these are the coefficient indices", model.coef_.nonzero())

    # naive version
    print("\n------ running naive OMP ------")
    candidates, budget = matching_pursuit(
        atoms=atoms,
        signal=signals[0],
        num_decomp=5,
        min_residual=0.001,
        is_omp=True,
        bandit_alg="",
        seed=0,
    )
    print("these are the coefficient indices", candidates)

    # bandit version
    print("\n------ running bandit OMP ------")
    candidates, budget = matching_pursuit(
        atoms=atoms,
        signal=signals[0],
        num_decomp=5,
        min_residual=0.001,
        is_omp=True,
        bandit_alg=ACTION_ELIMINATION,
        seed=0,
    )
    print("these are the coefficient indices", candidates)

