import numpy as np
import copy


def matching_pursuit(input_signal, atoms):
    """
    Canonical matching pursuit algorithm. Used for ground truth

    :param signal: Input vector to decompose into atoms
    :param atoms: Atoms with which to decompose signal into
    :return: vector of length len(atoms) with coefficients produced by matching pursuit
    """
    signal = copy.deepcopy(input_signal)
    counter = 0
    D = len(atoms)
    d = atoms.shape[1]
    coefs = np.zeros(D)
    assert np.allclose(np.linalg.norm(atoms, axis=1), np.ones(D))
    # Find maximum inner product atom
    norm_decreased = True
    while norm_decreased and np.linalg.norm(signal) > 0:
        max_inner_product = -float("inf")
        best_idx = -1
        for a_idx, a in enumerate(atoms):
            # NOTE: We get an extra count of d for atoms that have ALREADY been used
            ip = np.dot(signal, a)
            counter += d
            if ip > max_inner_product:
                max_inner_product = ip
                best_idx = a_idx

        # TODO: Need to implement signs for negative (look at ABOSOLUTE value of dot product)
        if max_inner_product > 0:
            coefs[best_idx] = max_inner_product
            signal -= coefs[best_idx]*atoms[best_idx]
        else:
            norm_decreased = False
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
    sigmas = np.std(atoms*signal_sampled, axis=1)  # Implicit broadcasting of *sampled_signal
    return sigmas


def b_matching_pursuit(input_signal, atoms, args):
    """
    Bandit version of matching pursuit

    :param signal: Input vector to decompose into atoms
    :param atoms: Atoms with which to decompose signal into
    :return: vector of length len(atoms) with coefficients produced by matching pursuit
    """
    signal = copy.deepcopy(input_signal)
    counter = 0

    D = len(atoms)
    d = atoms.shape[1]
    coefs = np.zeros(D)
    T = np.zeros(D)
    dims_sampled = 0
    threshold = 1e-2
    batch_size = 100
    batch_size = min(d, batch_size)
    if not args.p:
        p = 1. / (D * 1000)
    norm_decreased = True

    # NOTE: Assumes signal can be expressed in terms of atoms
    while norm_decreased and np.linalg.norm(signal) > 0:
        candidates = np.arange(D)
        lcbs = -np.inf * np.ones(D)
        ucbs = np.inf * np.ones(D)
        means = np.zeros(D)
        deltas = np.zeros(D)
        sigmas = get_sigmas(signal, atoms, args)
        # TODO: Add exact computation if total samples too big
        # TODO: Maybe need to change this to a >0 and ucbs > lcbs.max below
        while len(candidates) > 1:
            # For each remaining candidate:
            sample_dims = np.random.choice(np.arange(d), batch_size, replace=False)
            dims_sampled += batch_size
            signal_sampled = signal[sample_dims]
            candidates_sampled = atoms[candidates][:, sample_dims]  # NOTE: clever indexing because atoms[candidates,sample_dims] doesn't work
            # TODO: Need to perform compute_exactly condition, see https://github.com/motiwari/BanditPAM-python/blob/master/ucb_pam.py#L91-L106
            means[candidates] = \
                ((T[candidates] * means[candidates]) + (batch_size * (candidates_sampled @ signal_sampled))) / (batch_size + T[candidates])  # Does this np.dot do implicit broadcasting?
            T[candidates] += batch_size
            deltas[candidates] = sigmas[candidates] * np.sqrt(-np.log(p) / T[candidates])
            ucbs[candidates] = means[candidates] + deltas[candidates]
            lcbs[candidates] = means[candidates] - deltas[candidates]

            counter += len(candidates) * batch_size
            candidates = np.where((ucbs >= lcbs.max()))[0]  # NOTE: May have to change this to a > ?


        # TODO(@motiwari): There can be a problem here if the last candidate was computed exactly
        assert len(candidates) == 1  # Should be only one candidate remaining
        if ucbs[candidates][0] > 0:
            best_idx = candidates[0]
            coefs[best_idx] = np.dot(signal, atoms[best_idx])
            signal -= coefs[best_idx] * atoms[best_idx]
        else:
            norm_decreased = False
    return coefs, counter

