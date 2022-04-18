import copy
import numpy as np
import argparse
from typing import Tuple

from utils import write_log
from constants import NUM_STDS


def matching_pursuit(
    input_signal: np.ndarray,
    atoms: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, int]:
    """
    Canonical matching pursuit algorithm. Used for ground truth. Expected complexity is O(kd), where k = len(atoms)
    and d = len(input_signal).

    :param input_signal: Input vector to decompose into atoms
    :param atoms: Atoms with which to decompose signal into
    :return: vector of length len(atoms) with coefficients produced by matching pursuit
    """
    # stats will contain a set of lists to write to logfile.
    # Each child list contains the new norm of the signal, the index of the best atom, the coefficient of the residue with the best atom in this iteration, and the total coefficient of that atom (which may also contain coefficients from previous rounds)
    stats = []
    signal = copy.deepcopy(input_signal)
    counter = 0  # Number of coordinate-wise multiplications performed
    D = len(atoms)
    d = atoms.shape[1]
    coefs = np.zeros(D)
    assert np.allclose(np.linalg.norm(atoms, axis=1), np.ones(D))

    norm_decreased = True
    # while norm_decreased and np.linalg.norm(signal) > args.residue:  # Proceed until signal bas been fully expressed in terms of atoms.
    for _iter in range(args.max_iter):  # Perform the maximum number of iterations
        # Stop if the signal is ~0
        if np.linalg.norm(signal) < args.residue:
            break

        # Otherwise, find maximum inner product atom
        ips = np.abs(atoms @ signal)
        best_idx = np.argmax(ips)
        coef = np.dot(atoms[best_idx], signal)
        counter += D * d
        if np.abs(coef) > args.tol:
            coefs[best_idx] += coef
            signal -= coef * atoms[best_idx]
        else:
            norm_decreased = False
        stat = [np.linalg.norm(signal), best_idx, coef, coefs[best_idx]]
        stats.append(stat)
    write_log(
        args,
        stats,
        coefs,
        np.linalg.norm(input_signal),
        np.linalg.norm(signal),
        counter,
        bandit=False,
    )
    return coefs, counter


def get_batch_size(args: argparse.Namespace, d: int, min_value: int = 1000) -> int:
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = min_value
    return min(d, batch_size)


def get_sigmas(
    signal: np.ndarray,
    atoms: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """
    Get standard deviations for each arm returns. Works by sampling a batch of coordinates, computing the
    coordinate-wise inner product for thoose coordinates between the signal and each atom, and measuring the empirical
    standard deviation.

    Note: this can likely be replaced with the more statistsically correct empirical Bernstein inequality.

    :param signal: sampled signal (should be of dimension batch_size)
    :param atoms: sampled candidates (each should be of dimension batch_size)
    :param args: experiment arguments
    :return: standard eviations, the dispersion parameter for each arm, of length len(candidates_sampled)
    """
    d = atoms.shape[1]
    batch_size = get_batch_size(args, d)
    sample_dims = np.random.choice(np.arange(d), batch_size)
    signal_sampled = signal[sample_dims]

    # NOTE: EXTREMELY IMPORTANT: We multiply by a np.sqrt(d) here because the np.std(...) expression assumes
    # the r.v. under consideration is the COORDINATE-WISE product, whereas in the bandit_matching_pursuit algorithm,
    # the r.v. under consideration is the TOTAL inner product.
    sigmas = np.sqrt(d) * np.std(
        np.abs(atoms[:, sample_dims] * signal_sampled), axis=1
    )  # Implicit broadcasting of * sampled_signal
    return sigmas


def bandit_matching_pursuit(
    input_signal: np.ndarray,
    atoms: np.ndarray,
    args: argparse.Namespace,
    verbosity: int = 0,
) -> Tuple[np.ndarray, int]:
    """
    Bandit version of matching pursuit. Solves a best-arm identification problem in the multi-armed bandit framework.
    The arms correspond to different atoms. The arm returns correspond to the true inner product of each atom with the
    signal (or residual). Pulling an arm corresponds to computing a single coordinate's inner product with the signal
    (or residual).

    :param signal: Input vector to decompose into atoms
    :param atoms: Atoms with which to decompose signal into
    :return: vector of length len(atoms) with coefficients produced by matching pursuit
    """
    stats = []
    signal = copy.deepcopy(input_signal)
    counter = 0  # number of coordinate-wise multiplications performed
    D = len(atoms)
    d = atoms.shape[1]
    coefs = np.zeros(D)
    dims_sampled = 0
    batch_size = get_batch_size(args, d)

    if args.p is None:
        p = 1 / (D * d)
    norm_decreased = True

    # while norm_decreased and np.linalg.norm(signal) > 0:  # Terminate when signal is fully expressed in terms of atoms. NOTE: Assumes signal can be expressed in terms of atoms
    for _iter in range(args.max_iter):
        # Stop if the signal is ~0
        if np.linalg.norm(signal) < args.residue:
            break

        # Otherwise, run a UCB-like algorithm to find the best atom
        candidates = np.arange(D)  # arms (atoms)
        T = np.zeros(D)  # number of times we have fulled each "arm" (atom)
        lcbs = -np.inf * np.ones(D)  # lower confidence bounds of arm returns
        ucbs = np.inf * np.ones(D)  # upper confidence bounds of arm returns
        means = np.zeros(D)  # estimated means of arm returns
        deltas = np.zeros(
            D
        )  # the half-widths of the confidence intervals (ucb = mean + delta, lcb = mean - delta)
        exact_mask = np.zeros(
            D
        )  # a mask for which candidates have had their means (inner products) computed exactly
        sigmas = get_sigmas(
            signal, atoms, args
        )  # empirical estimates of standard deviations of arm returns

        # TODO: Add exact computation if total samples too big
        # TODO: Maybe need to change this to a >0 and ucbs > lcbs.max below
        # TODO: Don't let old candidates come back?
        # TODO: Change this to > 2, since if there is only one candidate left it will be the leader
        while len(candidates) > 1:

            # Figure out which arms would have more than d sample coordinates drawn in the next batch, and just compute
            # them exactly. This keeps the computation O(d) for arms that may be arbitrarily close to the leader.
            compute_exactly = np.where((T + batch_size >= d) & (exact_mask == 0))[0]
            if len(compute_exactly) > 0:
                means[compute_exactly] = np.abs(atoms[compute_exactly] @ signal)
                lcbs[compute_exactly] = ucbs[compute_exactly] = means[
                    compute_exactly
                ]  # The confidence interval contains only a single point
                exact_mask[compute_exactly] = 1
                T[compute_exactly] += d
                candidates = np.setdiff1d(
                    candidates, compute_exactly
                )  # Remove compute_exactly points from candidates so their bounds don't get updated below

            # TODO: Recompute candidates explicitly here with candidate condition to see if old, inexact arms overlap
            #  with a point that has just been computed exactly
            if len(candidates) == 0:
                break  # The last remaining candidates were computed exactly

            sample_dims = np.random.choice(np.arange(d), batch_size, replace=False)
            dims_sampled += batch_size
            signal_sampled = signal[sample_dims]
            candidates_sampled = atoms[candidates][
                :, sample_dims
            ]  # NOTE: clever indexing because atoms[candidates, sample_dims] doesn't work
            means[candidates] = (
                (T[candidates] * means[candidates])
                + (batch_size * np.abs(candidates_sampled @ signal_sampled))
            ) / (
                batch_size + T[candidates]
            )  # Update the running means of arm returns # Does this np.dot do implicit broadcasting?
            T[candidates] += batch_size
            deltas[candidates] = (
                NUM_STDS * sigmas[candidates] * np.sqrt(-np.log(p) / T[candidates])
            )
            ucbs[candidates] = means[candidates] + deltas[candidates]
            lcbs[candidates] = means[candidates] - deltas[candidates]

            candidates = np.where((ucbs >= lcbs.max()))[
                0
            ]  # The only remaining candidates are those whose CI overlap with the leader # NOTE: May have to change this to a > ?

        counter += np.sum(T)
        # Breaks exact ties with first. Also converts array to int.
        best_idx = np.arange(D)[np.where(means == means.max())][
            0
        ]  # TODO(@motiwari): Should this be mean?
        norm_decreased = False
        coef = np.dot(atoms[best_idx], signal)
        if (
            np.abs(coef) > args.tol
        ):  # Verify that the *actual* (not estimated) dot product is nontrivial. NOTE: This is an O(d) computation.
            coefs[best_idx] += coef
            signal -= coef * atoms[best_idx]
            norm_decreased = True
        stat = [np.linalg.norm(signal), best_idx, coef, coefs[best_idx]]
        stats.append(stat)
    counter = int(counter)  # The np.sum() promotes it to double above
    write_log(
        args,
        stats,
        coefs,
        np.linalg.norm(input_signal),
        np.linalg.norm(signal),
        int(counter),
        bandit=True,
    )
    return coefs, counter
