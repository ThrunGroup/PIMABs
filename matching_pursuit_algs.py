import copy
import numpy as np
from utils import get_logfile_name
from constants import NUM_STDS

def conv_to_str(input_):
    return ', '.join(map(str, map(lambda num: np.round(num, decimals=5), input_)))

def write_log(args, stats, coefs, init_signal_norm, final_signal_norm, total_cost, bandit):
    nonzero_coefs = list(np.where(np.abs(coefs) > 0)[0])
    nonzero_weights = list(coefs[nonzero_coefs])
    with open(get_logfile_name(args, bandit), 'w+') as fout:
        fout.write("Total cost: " + str(total_cost) + "\n")
        fout.write("Final answer:\n")
        fout.write("Nonzero indcs: " + conv_to_str(nonzero_coefs) + "\n")
        fout.write("Nonzero coefs: " + conv_to_str(nonzero_weights) + "\n")
        fout.write("Initial signal norm: " + str(init_signal_norm) + "\n")
        for row_i, row in enumerate(stats):
            fout.write("Iteration: " + str(row_i + 1) + "\n")
            fout.write(conv_to_str(row))
            fout.write("\n")
        fout.write("Final signal norm: " + str(final_signal_norm) + "\n")

def matching_pursuit(input_signal, atoms, args):
    """
    Canonical matching pursuit algorithm. Used for ground truth

    :param signal: Input vector to decompose into atoms
    :param atoms: Atoms with which to decompose signal into
    :return: vector of length len(atoms) with coefficients produced by matching pursuit
    """
    stats = []
    signal = copy.deepcopy(input_signal)
    counter = 0
    D = len(atoms)
    d = atoms.shape[1]
    coefs = np.zeros(D)
    assert np.allclose(np.linalg.norm(atoms, axis=1), np.ones(D))
    # Find maximum inner product atom
    norm_decreased = True
    #while norm_decreased and np.linalg.norm(signal) > 0:
    for _iter in range(args.max_iter):
        # Stop if the signal is ~0
        if np.linalg.norm(signal) < args.residue:
            break

        ips = np.abs(atoms @ signal)
        best_idx = np.argmax(ips)
        coef = np.dot(atoms[best_idx], signal)
        counter += D*d
        # TODO: Need to implement signs for negative (look at ABOSOLUTE value of dot product)
        if np.abs(coef) > args.tol:
            coefs[best_idx] += coef
            signal -= coef*atoms[best_idx]
        else:
            norm_decreased = False
        stat = [np.linalg.norm(signal), best_idx, coef, coefs[best_idx]]
        stats.append(stat)
    write_log(args, stats, coefs, np.linalg.norm(input_signal), np.linalg.norm(signal), counter, bandit=False)
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

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = round(.05*d)
    batch_size = min(d, batch_size)

    sample_dims = np.random.choice(np.arange(d), batch_size)
    signal_sampled = signal[sample_dims]
    # NOTE: EXTREMELY IMPORTANT: We multiply by a np.sqrt(d) here because the np.std(...) expression assumes
    # the r.v. under consideration is the COORDINATE-WISE product, whereas in the b_matching_pursuit algorithm,
    # the r.v. under consideration is the TOTAL inner product.
    sigmas = np.sqrt(d) * np.std(np.abs(atoms[:, sample_dims]*signal_sampled), axis=1)  # Implicit broadcasting of *sampled_signal
    return sigmas


def b_matching_pursuit(input_signal, atoms, args, verbosity=0):
    """
    Bandit version of matching pursuit

    :param signal: Input vector to decompose into atoms
    :param atoms: Atoms with which to decompose signal into
    :return: vector of length len(atoms) with coefficients produced by matching pursuit
    """
    stats = []
    signal = copy.deepcopy(input_signal)
    counter = 0
    D = len(atoms)
    d = atoms.shape[1]
    coefs = np.zeros(D)
    dims_sampled = 0

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = round(.05*d)
    batch_size = min(d, batch_size)

    if args.p is None:
        p = 1 / (D*d)
    norm_decreased = True

    # NOTE: Assumes signal can be expressed in terms of atoms
    #while norm_decreased and np.linalg.norm(signal) > 0:
    for _iter in range(args.max_iter):
        # Stop if the signal is ~0
        if np.linalg.norm(signal) < args.residue:
            break

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
                means[compute_exactly] = np.abs(atoms[compute_exactly] @ signal)
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
            deltas[candidates] = NUM_STDS*sigmas[candidates] * np.sqrt(-np.log(p) / T[candidates])
            ucbs[candidates] = means[candidates] + deltas[candidates]
            lcbs[candidates] = means[candidates] - deltas[candidates]

            candidates = np.where((ucbs >= lcbs.max()))[0]  # NOTE: May have to change this to a > ?

        counter += np.sum(T)
        # Breaks exact ties with first. Also converts array to int.
        best_idx = np.arange(D)[np.where(means == means.max())][0]  # TODO(@motiwari): Should this be mean?
        norm_decreased = False
        coef = np.dot(atoms[best_idx], signal)
        if np.abs(coef) > args.tol:  # Verify that the *actual* (not estimated) dot product is nontrivial
            coefs[best_idx] += coef
            signal -= coef * atoms[best_idx]
            norm_decreased = True
        stat = [np.linalg.norm(signal), best_idx, coef, coefs[best_idx]]
        stats.append(stat)
    counter = int(counter)  # The np.sum() promotes it to double above
    write_log(args, stats, coefs, np.linalg.norm(input_signal), np.linalg.norm(signal), int(counter), bandit=True)
    return coefs, counter

