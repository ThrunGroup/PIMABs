import numpy as np
import copy


def matching_pursuit(input_signal, atoms, args, verbosity=0):
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
        ips = np.abs(atoms @ signal)
        best_idx = np.argmax(ips)

        coef = np.dot(atoms[best_idx], signal)
        counter += D*d
        # TODO: Need to implement signs for negative (look at ABOSOLUTE value of dot product)
        if np.abs(coef) > args.tol:
            coefs[best_idx] = coef
            signal -= coef*atoms[best_idx]
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
    sigmas = np.std(np.abs(atoms[:, sample_dims]*signal_sampled), axis=1)  # Implicit broadcasting of *sampled_signal
    return sigmas


def b_matching_pursuit(input_signal, atoms, args, verbosity=0):
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
    dims_sampled = 0
    batch_size = min(d, args.batch_size)
    if not args.p:
        p = 1. / (D * 1000)
    norm_decreased = True

    # NOTE: Assumes signal can be expressed in terms of atoms
    while norm_decreased and np.linalg.norm(signal) > 0:
        candidates = np.arange(D)
        T = np.zeros(D)
        lcbs = -np.inf * np.ones(D)
        ucbs = np.inf * np.ones(D)
        means = np.zeros(D)
        deltas = np.zeros(D)
        exact_mask = np.zeros(D)
        sigmas = get_sigmas(signal, atoms, args)

        # TODO: Add exact computation if total samples too big
        # TODO: Maybe need to change this to a >0 and ucbs > lcbs.max below
        while len(candidates) > 1:
            compute_exactly = np.where((T + batch_size >= d) & (exact_mask == 0))[0]
            if len(compute_exactly) > 0:
                means[compute_exactly] = atoms[compute_exactly] @ signal
                lcbs[compute_exactly] = ucbs[compute_exactly] = means[compute_exactly]
                exact_mask[compute_exactly] = 1
                T[compute_exactly] += d
                candidates = np.setdiff1d(candidates, compute_exactly)  # Remove compute_exactly points from candidates so they're bounds don't get updated below

            if len(candidates) == 0:
                break  # The last remaining candidates were computed exactly

            sample_dims = np.random.choice(np.arange(d), batch_size, replace=False)
            dims_sampled += batch_size
            signal_sampled = signal[sample_dims]
            candidates_sampled = atoms[candidates][:, sample_dims]  # NOTE: clever indexing because atoms[candidates,sample_dims] doesn't work
            means[candidates] = \
                ((T[candidates] * means[candidates]) + (batch_size * np.abs(candidates_sampled @ signal_sampled))) / (batch_size + T[candidates])  # Does this np.dot do implicit broadcasting?
            T[candidates] += batch_size
            deltas[candidates] = sigmas[candidates] * np.sqrt(-np.log(p) / T[candidates])
            ucbs[candidates] = means[candidates] + deltas[candidates]
            lcbs[candidates] = means[candidates] - deltas[candidates]

            candidates = np.where((ucbs >= lcbs.max()))[0]  # NOTE: May have to change this to a > ?

        counter += np.sum(T)
        # Breaks exact ties with first. Also converts array to int.
        best_idx = np.arange(D)[np.where(ucbs == ucbs.max())][0]  # TODO(@motiwari): Should this be mean?

        norm_decreased = False
        coef = np.dot(atoms[best_idx], signal)
        if np.abs(coef) > args.tol:  # Verify that the *actual* (not estimated) dot product is nontrivial
            coefs[best_idx] = coef
            signal -= coef * atoms[best_idx]
            norm_decreased = True

    return coefs, counter

