from sklearn.linear_model import OrthogonalMatchingPursuit  # for debugging purposes
from data.get_data import get_data
from utils.utils import set_seed
from utils.constants import (
    NORMAL_CUSTOM,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
)

import numpy as np
from math import sqrt
from scipy import linalg
from typing import Tuple
from scipy.linalg.lapack import get_lapack_funcs

from algorithms.mips_bandit import mips_bandit

def matching_pursuit(
        atoms: np.ndarray,
        signal: np.ndarray,
        n_nonzero_coefs: int = 5,
        min_residual: float = 0.001,
        bandit_alg: str = "",
        seed: int = 0,
        abs: bool = False,
        tol=None,
):
    """
    Runs the Matching Pursuit algorithm. Refer to implementation by https://github.com/stonemason11
    :param atoms: Atoms array
    :param signal: Signal vector
    :param n_nonzero_coefs: number of decompositions we want to make (equivalent to scikit-learn's n_nonzero_coefs)
    :param min_residual: the threshold for which to terminate
    :param bandit_alg: Name of the bandit algorithm to get MIPS. Default naive.
    :param seed: Seed
    :return: an array of indices similar to scikit-learn OMP's indices attribute AND budget
    """
    set_seed(seed)
    decomp = 0
    total_budget = 0
    unexplored = np.arange(atoms.shape[0])
    candidates = np.zeros(n_nonzero_coefs, dtype=np.int64)
    residual = signal

    while decomp < n_nonzero_coefs and la.norm(residual) > min_residual:
        # solve the MIPS problem
        if len(bandit_alg) > 0:
            candidate, budget = mips_bandit(
                atoms[unexplored],  # exclude previous decompositions
                np.expand_dims(residual, axis=0),  # need to pass in 2d array
                maxmin=(np.min(atoms), np.max(atoms)),
                bandit_alg=bandit_alg,
                epsilon=0.01,
                abs=abs,
                delta=0.01,
            )
        else:
            if abs:
                candidate = np.argmax(np.abs(atoms[unexplored] @ residual))
            else:
                candidate = np.argmax(atoms[unexplored] @ residual)
            budget = atoms[unexplored].shape[0] * atoms[unexplored].shape[1]

        # update candidates and exclude them from atoms
        candidates[decomp] = candidate
        unexplored = np.delete(unexplored, candidate, axis=0)
        total_budget += budget
        decomp += 1

        # update the residual
        residual -= (residual @ atoms[candidate]) * atoms[candidate]

    return candidates, total_budget


def orthogonal_matching_pursuit(
        atoms: np.ndarray,
        signal: np.ndarray,
        n_nonzero_coefs: int = 5,
        min_residual: float = 0.001,
        bandit_alg: str = "",
        seed: int = 0,
        abs: bool = False,
        tol=None,
) -> Tuple[np.ndarray, int]:
    """
    Runs the Orthogonal Matching Pursuit algorithm.
    Most of the codes were borrowed from scikit-learn
    See https://github.com/scikit-learn/scikit-learn/blob/36958fb240fbe435673a9e3c52e769f01f36bec0/sklearn/linear_model/_omp.py#L29

    :param atoms: Atoms array
    :param signal: Signal vector
    :param n_nonzero_coefs: number of decompositions we want to make (equivalent to scikit-learn's n_nonzero_coefs)
    :param min_residual: the threshold for which to terminate
    :param bandit_alg: Name of the bandit algorithm to get MIPS. Default naive.
    :param seed: Seed
    :return: an array of indices similar to scikit-learn OMP's indices attribute AND budget
    """
    set_seed(seed)

    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    atoms = atoms.copy("F")

    total_budget = 0  # BANDIT
    unexplored = np.arange(atoms.T.shape[0])  # BANDIT

    min_float = np.finfo(atoms.dtype).eps
    nrm2, swap = linalg.get_blas_funcs(("nrm2", "swap"), (atoms,))
    (potrs,) = get_lapack_funcs(("potrs",), (atoms,))

    alpha = np.dot(atoms.T, signal)
    residual = signal
    n_active = 0
    indices = np.arange(atoms.shape[1])  # keeping track of swapping

    max_features = atoms.shape[1] if tol is not None else n_nonzero_coefs

    L = np.empty((max_features, max_features), dtype=atoms.dtype)

    while np.linalg.norm(residual) > min_residual:
        # solve the MIPS problem
        if len(bandit_alg) > 0:
            candidate, budget = mips_bandit(
                atoms.T[unexplored],  # exclude previous decompositions
                residual.T,  # need to pass in 2d array
                maxmin=(np.min(atoms), np.max(atoms)),
                bandit_alg=bandit_alg,
                epsilon=0.0001,
                abs=abs,
                delta=0.00001,
            )
            lam = candidate[0][0]
            unexplored = np.delete(unexplored, candidate, axis=0)
        else:
            # import ipdb; ipdb.set_trace()
            if abs:
                lam = np.argmax(np.abs(atoms.T @ residual))
            else:
                lam = np.argmax((atoms.T @ residual))
            budget = atoms.shape[0] * atoms.shape[1]

        total_budget += budget

        if lam < n_active or alpha[lam] ** 2 < min_float:
            # atom already selected or inner product too small
            print("atom already selected")
            break

        if n_active > 0:
            # Updates the Cholesky decomposition of atoms' atoms
            L[n_active, :n_active] = np.dot(atoms[:, :n_active].T, atoms[:, lam])
            linalg.solve_triangular(
                L[:n_active, :n_active],
                L[n_active, :n_active],
                trans=0,
                lower=1,
                overwrite_b=True,
                check_finite=False,
            )
            v = nrm2(L[n_active, :n_active]) ** 2
            Lkk = linalg.norm(atoms[:, lam]) ** 2 - v
            if Lkk <= min_float:  # selected atoms are dependent
                print("selected atoms are dependent")
                break
            L[n_active, n_active] = sqrt(Lkk)
        else:
            L[0, 0] = linalg.norm(atoms[:, lam])

        atoms.T[n_active], atoms.T[lam] = swap(atoms.T[n_active], atoms.T[lam])
        alpha[n_active], alpha[lam] = alpha[lam], alpha[n_active]
        indices[n_active], indices[lam] = indices[lam], indices[n_active]
        n_active += 1

        # solves LL'atoms = atoms'signal as a composition of two triangular systems
        gamma, _ = potrs(
            L[:n_active, :n_active], alpha[:n_active], lower=True, overwrite_b=False
        )

        residual = signal - np.dot(atoms[:, :n_active], gamma)
        if tol is not None and nrm2(residual) ** 2 <= tol:
            break
        elif n_active == max_features:
            break

    indices = indices[:n_active]
    # indices.sort()
    return indices, total_budget


if __name__ == "__main__":

    for seed in range(10):
        print("\nSeed : ", seed)

        atoms, signals = get_data(
            num_atoms=200,
            len_signal=2000,
            seed=seed,
            data_type=NORMAL_CUSTOM,
            num_signals=1,
        )

        # scikit-learn
        model1 = OrthogonalMatchingPursuit(normalize=False, n_nonzero_coefs=5, fit_intercept=False, precompute=False)
        print("------- running scikit-learn OMP ---------")
        model1.fit(atoms.T, signals[0])
        print("these are the coefficient indices (sorted by default)", model1.coef_.nonzero())

        # naive version
        print("------ running naive OMP ------")
        candidates, budget = orthogonal_matching_pursuit(
            atoms=atoms.T,
            signal=signals[0],
            n_nonzero_coefs=5,
            min_residual=0.001,
            bandit_alg="",
            abs=True,
            seed=0,
        )
        print("these are the coefficient indices", candidates)

        # bandit version
        print("\n------ running bandit OMP ------")
        candidates, budget = orthogonal_matching_pursuit(
            atoms=atoms.T,
            signal=signals[0],
            n_nonzero_coefs=5,
            min_residual=0.001,
            bandit_alg=ACTION_ELIMINATION,
            abs=True,
            seed=0,
        )
        print("these are the coefficient indices", candidates)