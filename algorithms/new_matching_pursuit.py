from sklearn.linear_model import OrthogonalMatchingPursuit  # for debugging purposes
from data.get_data import get_data
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

def orthogonal_matching_pursuit(
    atoms: np.ndarray,
    signal: np.ndarray,
    n_nonzero_coefs: int = 5,
    min_residual: float = 0.001,
    is_omp: bool = False,
    bandit_alg: str = "",
    seed: int = 0,
    abs: bool = False,
    copy_X=True,
    return_path=False,
    tol=None,
) -> Tuple[np.ndarray, int]:
    """
    Most of the codes were copied from scikit-learn
    See https://github.com/scikit-learn/scikit-learn/blob/36958fb240fbe435673a9e3c52e769f01f36bec0/sklearn/linear_model/_omp.py#L29
    """
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    if copy_X:
        atoms = atoms.copy("F")
    else:  # even if we are allowed to overwrite, still copy it if bad order
        atoms = np.asfortranarray(atoms)

    total_budget = 0  # BANDIT
    unexplored = np.arange(atoms.T.shape[0]) # BANDIT

    min_float = np.finfo(atoms.dtype).eps
    nrm2, swap = linalg.get_blas_funcs(("nrm2", "swap"), (atoms,))
    (potrs,) = get_lapack_funcs(("potrs",), (atoms,))

    alpha = np.dot(atoms.T, signal)
    residual = signal
    n_active = 0
    indices = np.arange(atoms.shape[1])  # keeping track of swapping

    max_features = atoms.shape[1] if tol is not None else n_nonzero_coefs

    L = np.empty((max_features, max_features), dtype=atoms.dtype)

    if return_path:
        coefs = np.empty_like(L)

    while True:
        # lam = np.argmax(np.abs(np.dot(atoms.T, residual)))
        # solve the MIPS problem
        if len(bandit_alg) > 0:
            candidate, budget = mips_bandit(
                atoms.T[unexplored],  # exclude previous decompositions
                residual.T,  # need to pass in 2d array
                # np.expand_dims(residual[0], axis=0),  # need to pass in 2d array
                maxmin=(-3.7, 3.7),
                bandit_alg=bandit_alg,
                epsilon=0.01,
                abs=abs,
                delta=0.01,
            )
            lam = candidate[0][0]
            unexplored = np.delete(unexplored, candidate, axis=0)
        else:
            # import ipdb; ipdb.set_trace()
            if abs:
                # print(np.abs(atoms.T @ residual))
                lam = np.argmax(np.abs(atoms.T @ residual))
            else:
                lam = np.argmax((atoms.T @ residual))
            budget = atoms.shape[0] * atoms.shape[1]

        total_budget += budget

        print(n_active, lam)

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

        if return_path:
            coefs[:n_active, n_active - 1] = gamma
        residual = signal - np.dot(atoms[:, :n_active], gamma)
        if tol is not None and nrm2(residual) ** 2 <= tol:
            break
        elif n_active == max_features:
            break

    indices = indices[:n_active]
    indices.sort()
    return indices, total_budget

if __name__ == "__main__":
    atoms, signals = get_data(
        num_atoms=100,
        len_signal=1000,
        seed=0,
        data_type=NORMAL_CUSTOM,
        num_signals=1,
    )

    from sklearn.datasets import make_regression
    atoms, signals = make_regression(noise=4, random_state=0)
    signals = [signals]

    print(np.min(atoms), np.max(atoms))

    # scikit-learn
    model1 = OrthogonalMatchingPursuit(normalize=False, n_nonzero_coefs=3, fit_intercept=False, precompute=False)
    print("------- running scikit-learn OMP (fit_interncept=false) ---------")
    model1.fit(atoms.T, signals[0])
    print("these are the coefficient indices", model1.coef_.nonzero())

    # naive version
    print("\n------ running naive OMP ------")
    candidates, budget = orthogonal_matching_pursuit(
        atoms=atoms.T,
        signal=signals[0],
        n_nonzero_coefs=3,
        min_residual=0.001,
        is_omp=True,
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
        n_nonzero_coefs=3,
        min_residual=0.001,
        is_omp=True,
        bandit_alg=ACTION_ELIMINATION,
        abs=True,
        seed=0,
    )
    print("these are the coefficient indices", candidates)