import numpy as np
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Solve MIPS problem")
    parser.add_argument("--get_ground_truth", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--seed", default=0)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--p", type=float)
    return parser.parse_args()

def counter_function():
    pass

def matching_pursuit(signal, atoms):
    """
    Canonical matching pursuit algorithm. Used for ground truth

    :param signal: Input vector to decompose into atoms
    :param atoms: Atoms with which to decompose signal into
    :return: vector of length len(atoms) with coefficients produced by matching pursuit
    """
    counter = 0
    D = len(atoms)
    d = atoms.shape[1]
    coefs = np.zeros(D)
    assert np.allclose(np.linalg.norm(atoms, axis=1), np.ones(D))
    threshold = 1e-2
    # Find maximum inner product atom
    while np.linalg.norm(signal) > threshold:
        max_inner_product = -float("inf")
        best_idx = -1

        for a_idx, a in enumerate(atoms):
            # NOTE: We get an extra count of d for atoms that have ALREADY been used
            ip = np.dot(signal, a)
            counter += d
            if ip > max_inner_product:
                max_inner_product = ip
                best_idx = a_idx

        coefs[best_idx] = max_inner_product
        signal -= coefs[best_idx]*atoms[best_idx]
    return coefs, counter

def get_sigmas(signal, atoms, args):
    """
    Gets sigmas for confidence bound computation

    :param signal_sampled: Sampled signal (should be of dimension batch_size)
    :param candidates_sampled: Sampled candidates (each should be of dimension batch_size)
    :param args: Experiment arguments
    :return: Sigmas, the dispersion parameter for each arm, of length len(candidates_sampled)
    """
    d = atoms.shape[1]
    batch_size = min(d, args.batch_size)
    sample_dims = np.random.choice(np.arange(d), batch_size)
    signal_sampled = signal[sample_dims]
    atoms_sampled = atoms[:, sample_dims]
    sigmas = np.std(atoms*signal_sampled, axis=1)  # Implicit broadcasting of *sampled_signal
    return sigmas


def b_matching_pursuit(signal, atoms, args):
    """
    Bandit version of matching pursuit

    :param signal: Input vector to decompose into atoms
    :param atoms: Atoms with which to decompose signal into
    :return: vector of length len(atoms) with coefficients produced by matching pursuit
    """
    counter = 0

    D = len(atoms)
    d = atoms.shape[1]
    coefs = np.zeros(D)
    T = np.zeros(D)
    sigmas = np.zeros(D)
    dims_sampled = 0
    threshold = 1e-2
    batch_size = 100
    batch_size = min(d, batch_size)
    if not args.p:
        p = 1. / D * 1000

    while np.linalg.norm(signal) > threshold:
        candidates = np.arange(D)
        lcbs = -np.inf() * np.ones(D)
        ucbs = np.inf() * np.ones(D)
        means = np.zeros(D)
        deltas = np.zeros(D)
        sigmas = get_sigmas(signal, atoms, args)
        # TODO: Add exact computation if total samples too big
        while len(candidates) > 1:
            # For each remaining candidate:
            sample_dims = np.random.choice(np.arange(d), batch_size)
            signal_sampled = signal[sample_dims]
            candidates_sampled = atoms[candidates, sample_dims]
            # TODO: Broadcast this properly, see https://github.com/motiwari/BanditPAM-python/blob/master/ucb_pam.py#L129-L134
            for c_idx, c in enumerate(candidates_sampled):
                # TODO: Need to perform compute_exactly condition, see https://github.com/motiwari/BanditPAM-python/blob/master/ucb_pam.py#L91-L106
                orig_idx = candidates[c_idx]
                means[orig_idx] = \
                    ((T[orig_idx] * means[orig_idx]) + (batch_size * np.dot(signal_sampled, c))) / (batch_size + T[orig_idx])  # Does this np.dot do implicit broadcasting?
                T[orig_idx] += batch_size
                deltas[orig_idx] = sigmas[orig_idx] * np.sqrt(-np.log(args.p) / T[orig_idx])
                ucbs[orig_idx] = means[orig_idx] - deltas[orig_idx]
                lcbs[orig_idx] = means[orig_idx] + deltas[orig_idx]

            candidates = np.where((lcbs < ucbs.min()))[0]
            counter += len(candidates)*batch_size
            # filter candidates

        assert len(candidates) == 1  # Should be only one candidate remaining
        best_idx = candidates[0]
        coefs[best_idx] = np.dot(signal, atoms[best_idx])
        signal -= coefs[best_idx] * atoms[best_idx]
    return coefs, counter

def main(args):
    np.random.seed(args.seed)
    test_case_1()
    test_case_2()
    test_case_3()


def test_case_1():
    dim = 5
    signal = np.ones(dim)
    atoms = np.array([
        np.ones(dim) / np.sqrt(5),
        [0, 1, 0, 0, 1] / np.sqrt(2),
    ])
    coefs, counter = matching_pursuit(signal, atoms)
    assert np.allclose(coefs, np.array([2.23606798, 0]))
    assert counter == 10

def test_case_2():
    signal = np.array([1., 0, 1, 0, 1, 0, 1])
    atoms = np.array([
        np.array([1., 0, 1, 0, 1, 0, 1]) / 2,
        np.array([0, 1., 0, 1, 0, 1, 0]) / np.sqrt(3),
    ])
    coefs, counter = matching_pursuit(signal, atoms)
    assert np.allclose(coefs, np.array([2, 0]))
    assert counter == 14

def test_case_3():
    signal = np.array([1., 1, 1, 0, 1, 0, 1])
    atoms = np.array([
        np.array([1., 0, 1, 0, 1, 0, 1]) / 2,
        np.array([0, 1., 0, 1, 0, 1, 0]) / np.sqrt(3),
        np.array([0, 1., 0, 0, 0, 0, 0]),
    ])
    coefs, counter = matching_pursuit(signal, atoms)
    assert np.allclose(coefs, np.array([2, 0, 1]))
    assert counter == 42


if __name__ == '__main__':
    args = get_args()
    print(main(args))