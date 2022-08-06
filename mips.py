import numpy as np
import math
from typing import Tuple

from constants import (
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


def action_elimination(
    atoms: np.ndarray,
    signal: np.ndarray,
    var_proxy: np.ndarray,
    epsilon: float,
    delta: float,
    num_best_atoms: int = 1,
    verbose: bool = False,
    batch_size: int = BATCH_SIZE,
) -> Tuple[np.ndarray, int]:
    """
    A best-arm identification algorithm that eliminates the arms whose upper bound is less than the lower bound of the
    arm with highest estimated reward in each round. See algorithm 1 in https://arxiv.org/pdf/2006.06856.pdf and
    AE algorithm in https://homes.cs.washington.edu/~jamieson/resources/bestArmSurvey.pdf#page=2

    :return: Returns the k(num_best_atoms) best arms with error epsilon and probability 1 - delta.
    """
    D = atoms.shape[0]  # Number of atoms
    d = atoms.shape[1]  # Dimension of atoms/signal
    t = 0  # time step
    new_delta = delta / (d * D)
    budgets = 0  # Number of calculations
    candidates = np.arange(D)  # arms (atoms)
    lcbs = -np.inf * np.ones(D)  # lower confidence bounds of arm returns
    ucbs = np.inf * np.ones(D)  # upper confidence bounds of arm returns
    means = np.zeros(D)  # estimated means of arm returns
    # TODO: Add exact computation if total mnist too big
    # TODO: Maybe need to change this to a >0 and ucbs > lcbs.max below
    # TODO: Don't let old candidates come back?
    # TODO: Change this to > 2, since if there is only one candidate left it will be the leader
    while len(candidates) > num_best_atoms:
        t += 1

        # Find a true inner product value of candidates if we sample more than their dimension (length of atom/signal)
        if t * batch_size > d:
            means[candidates] = np.abs(atoms[candidates] @ signal)
            budgets += d * len(candidates)
            return candidates[np.argsort(-means[candidates])][:num_best_atoms], budgets
        budgets += batch_size * len(candidates)

        sample_idcs = np.random.choice(np.arange(d), batch_size, replace=False)
        signal_sampled = signal[sample_idcs]
        candidates_sampled = subset_2d(atoms, rows=candidates, columns=sample_idcs)
        means[candidates] = (
            (t * batch_size * means[candidates]) + (candidates_sampled @ signal_sampled)
        ) / (
            (t + 1) * batch_size
        )  # Update the running means of arm returns # Does this np.dot do implicit broadcasting?
        ci = get_ci(
            delta=new_delta,
            var_proxy=var_proxy,
            ci_bound=HOEFFDING,
            num_samples=t * batch_size,
        )
        if 4 * ci < epsilon:
            return candidates[np.argsort(-means[candidates])][:num_best_atoms], budgets
        ucbs[candidates] = means[candidates] + ci
        lcbs[candidates] = means[candidates] - ci
        new_cand_idcs = np.where((ucbs[candidates] >= lcbs[candidates].max()))[0]

        if (
            len(new_cand_idcs) < num_best_atoms
        ):  # Have to return candidates with length num_best_atoms
            return candidates[np.argsort(-means[candidates])][:num_best_atoms], budgets
        candidates = candidates[new_cand_idcs]
    return candidates, budgets


def adaptive_action_elimination(
    atoms: np.ndarray,
    signal: np.ndarray,
    var_proxy: np.ndarray,
    epsilon: float,
    delta: float,
    num_best_atoms: int = 1,
    verbose: bool = False,
    batch_size: int = BATCH_SIZE,
) -> Tuple[np.ndarray, int]:
    """
    An adaptive AE(Action Elimination) algorithm that adopts sampling without replacement, adaptive variance proxy, and
    smart sampling on top of original AE algorithm. It applies to MIPS(Maximum Inner product search) problem adaptively.

    Adaptive changes
    -Sample without replacement: Get a tight confidence interval by sampling without replacement, see lemma 1 in
    https://arxiv.org/pdf/1309.4029.pdf
    -Adaptive variance proxy: Tracking the sampled elements of signal vector and atoms, we can make the bounds of
    arms' rewards(dot product of atom and signal whose elements are sampled) tighter.
    -Smart sampling: By sorting the signal array, we can make confidence interval tighter and can expect the faster
    best arm identification.

    Should assert that all vectors are normalized. Omit this assertion statement in function due to its hevay
    computation.

    :return: Returns the k(num_best_atoms) best arms with error epsilon and probability 1 - delta.
    """
    D = atoms.shape[0]  # Number of atoms
    d = atoms.shape[1]  # Dimension of atoms/signal
    t = 0  # time step
    new_delta = delta / (d * D)
    budgets = 0  # Number of calculations
    candidates = np.arange(D)  # arms (atoms)
    lcbs = -np.inf * np.ones(D)  # lower confidence bounds of arm returns
    ucbs = np.inf * np.ones(D)  # upper confidence bounds of arm returns
    means = np.zeros(D)  # estimated means of arm returns
    sorted_idcs = np.argsort(
        -np.abs(signal)
    )  # Indices that sorts signal in the descending order
    l2_atoms = np.zeros(D)  # l2 norms of atoms

    while len(candidates) > num_best_atoms:
        t += 1

        # Find a true inner product value of candidates if we sample as much as their length (length of atom/signal)
        if t * batch_size > d:
            means[candidates] = np.abs(atoms[candidates] @ signal)
            budgets += d * len(candidates)
            return candidates[np.argsort(-means[candidates])][:num_best_atoms], budgets

        batch_size = min(
            batch_size, d - (t - 1) * batch_size
        )  # Can't sample more than population size
        sample_idcs = sorted_idcs[(t - 1) * batch_size : t * batch_size]
        signal_sampled = signal[sample_idcs]
        candidates_sampled = subset_2d(atoms, rows=candidates, columns=sample_idcs)
        means[candidates] = (
            (t * batch_size * means[candidates]) + (candidates_sampled @ signal_sampled)
        ) / ((t + 1) * batch_size)

        # Make confidence interval tighter using the constraint that all the vectors have unit length.
        l2_atoms[candidates] += np.sum(
            np.square(candidates_sampled), axis=1
        )  # l2 norm of sampled atoms
        max_signal = np.abs(signal_sampled[0])  # maximum of signal sampled
        multiplication_bounds = (
            2 * (1 - l2_atoms[candidates]) * np.abs(max_signal)
        )  # The bound of the multiplication
        # between each element from signal and atoms
        new_var_proxy = (
            multiplication_bounds ** 2 / 4
        )  # See Hoeffding's lemma (lemma 18) in https://bit.ly/3RXawav
        new_var_proxy = np.minimum(var_proxy, new_var_proxy)
        ci_vec = get_ci(
            delta=new_delta,
            var_proxy=new_var_proxy,
            ci_bound=HOEFFDING,
            num_samples=t * batch_size,
        )
        budgets += batch_size * len(candidates)
        if np.all(4 * ci_vec < epsilon):
            return candidates[np.argsort(-means[candidates])][:num_best_atoms], budgets
        ucbs[candidates] = means[candidates] + ci_vec
        lcbs[candidates] = means[candidates] - ci_vec
        new_cand_idcs = np.where((ucbs[candidates] >= lcbs[candidates].max()))[0]

        if (
            len(new_cand_idcs) < num_best_atoms
        ):  # Have to return candidates with length num_best_atoms
            return candidates[np.argsort(-means[candidates])][:num_best_atoms], budgets
        candidates = candidates[new_cand_idcs]
    return candidates, budgets


def median_elimination(
    atoms: np.ndarray,
    signal: np.ndarray,
    epsilon: float,
    delta: float,
    num_best_atoms: int = 1,
    verbose: bool = False,
    is_experiment: bool = False,
    seed: int = 0,
) -> Tuple[np.ndarray, int]:
    """
    Median Elimination algorithm + Sampling without replacement, see algorithm 1 in shorturl.at/adPR3

    :return: Returns the k(num_best_atoms) best arms with error epsilon and probability 1 - delta.
    """
    rng = np.random.default_rng(seed)
    len_signal = signal.shape[0]
    candidates = np.arange(atoms.shape[0])
    estimates = np.zeros(atoms.shape[0])
    population_idcs = np.arange(len_signal)
    new_epsilon = epsilon / 4
    new_delta = delta / 2
    prev_T = 0  # A helper variable for calculating the optimal number of mnist in each round
    total_num_samples = 0
    budget = 0  # Total number of calculations

    # U = (math.log(1 / delta) * (maxmin[0] - maxmin[1]) ** 2) / (2 * epsilon ** 2)
    def m(u):
        """
        A helper function for calculating the optimal number of mnist
        """
        return min(
            (u + 1) / (1 + u / len_signal), (u + u / len_signal) / (1 + u / len_signal),
        )

    while len(candidates) > num_best_atoms:
        # Calculate the optimal number of mnist in the current round
        new_T = m(
            2
            * math.log(
                2
                * (len(candidates) - num_best_atoms)
                / (
                    new_delta
                    * (math.floor(((len(candidates)) - num_best_atoms) / 2) + 1)
                )
            )
            / (new_epsilon ** 2)
        )
        current_num_samples = int(new_T - prev_T)
        prev_T = new_T
        budget += current_num_samples * len(candidates)

        # Sample without replacement using Fisher-Yates shuffle.
        # Time complexity is linear in len(current_num_samples)
        if not is_experiment:
            idcs = rng.choice(
                len_signal - total_num_samples, current_num_samples, replace=False,
            )
        else:
            # With an adversarial dataset, wan do a worst-case experiment by not shuffling a data
            # Ex) atoms = np.transpose(np.array([[0, 0, 0, 0, 1, 1, 1, 1],
            #                                         [1, 1, 1, 1, 0, 0, 0, 0]])
            #     signal = np.ones(8)
            #
            #     In this case, sampling without shuffling will cause a poor performance since
            #     a first few elements of each atom doesn't represent their distribution well.
            #     So, it's a worst case scenario.
            idcs = np.arange(current_num_samples, dtype=np.int64)
        sample_idcs = population_idcs[idcs]
        population_idcs = np.delete(population_idcs, idcs)

        estimates[candidates] = (
            estimates[candidates] * total_num_samples
            + (np.matmul(atoms[:, sample_idcs][candidates], signal[sample_idcs]))
        ) / (total_num_samples + current_num_samples)

        candidates = candidates[
            np.argsort(estimates[candidates])
        ]  # sort candidates with estimates
        candidates = candidates[math.ceil((len(candidates) - num_best_atoms) / 2) :]
        new_epsilon *= 3 / 4
        new_delta *= 1 / 2
        total_num_samples += current_num_samples
    if verbose:
        print(f"budget is {budget}")
    return candidates, budget


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


def fixed_confidence_exp(
    epsilons: Tuple[float],
    deltas: Tuple[float],
    generate_data: str = ADVERSARIAL_CUSTOM,
    bandit_alg: str = MEDIAN_ELIMINATION,
    num_experiments: int = 20,
    verbose: bool = False,
    with_replacement: bool = True,
) -> None:
    """
    Validate that bandit algorithm for fixed confidence setting works well. In other words, observed failure
    probability for bandit algorithm with fixed confidence setting should be lower than failure probability(delta)
    passed as a parameter for a bandit algorithm.

    :param epsilons: A tuple of epsilons
    :param deltas: A tuple of deltas
    :param generate_data: A name of data generating function
    :param bandit_alg: A name of bandit algorithm
    :param num_experiments: Number of experiments for each fixed confidence setting.
    :param verbose: verbosity of function
    :param with_replacement: Whether the algorithm mnist with replacement
    """
    if generate_data == ADVERSARIAL_CUSTOM:
        generate_data = generate_adversarial_data
        maxmin = (1, 0)
    elif generate_data == NORMAL_CUSTOM:
        generate_data = generate_normal_data
    else:
        raise NotImplementedError(f"{generate_data} is not implemented")
    for epsilon in epsilons:
        for delta in deltas:
            failure_counts = 0
            print(f"Doing an experiment with epsilon={epsilon} and delta={delta}")
            for seed in range(num_experiments):
                if verbose:
                    print(f"-seed: {seed}")
                atoms, signal = generate_data(seed=seed)
                if verbose:
                    print("--data generated")
                true_inner_products = np.matmul(atoms, signal)
                true_best_idx = np.argmax(true_inner_products)
                if verbose:
                    print("--naively solved")
                estimated_best_idx, _ = mips_bandit(
                    atoms=atoms,
                    signal=signal,
                    maxmin=maxmin,
                    bandit_alg=bandit_alg,
                    epsilon=epsilon,
                    delta=delta,
                    num_best_atoms=1,
                    seed=seed,
                    is_experiment=False,
                    with_replacement=with_replacement,
                )
                if verbose:
                    print("--stochastically solved")
                best_arm_mean = true_inner_products[true_best_idx] / signal.shape[0]
                estimated_best_arm_mean = (
                    true_inner_products[estimated_best_idx][0] / signal.shape[0]
                )
                if abs(best_arm_mean - estimated_best_arm_mean) > epsilon:
                    failure_counts += 1
            observed_delta = failure_counts / num_experiments
            print(f"delta: {delta}, epsilon: {epsilon}")
            print(f"observed failure probability: {observed_delta}")
            print(f"upper bound of failure probability: {delta}\n")
            assert (
                observed_delta < delta
            ), f"{bandit_alg} is theoretically wrong in fixed confidence setting"


def scaling_exp(
    epsilon: float,
    delta: float,
    generate_data: str = ADVERSARIAL_CUSTOM,
    bandit_alg: str = MEDIAN_ELIMINATION,
    num_experiments: int = 20,
    size_minmax: Tuple[int] = (10, 10 ** 6),
    seed: int = 0,
    with_replacement: bool = False,
    is_normalize: bool = True,
) -> None:
    """
    Check that the time complexity of bandit algorithm scales with the square root of the signal size.
    MEDIAN_ELIMINATION algorithm should satisfy this scaling according to Corollary 3 in "shorturl.at/adPR3".

    :param epsilon: A epsilon (an additive error allowed)
    :param delta: A delta (an upper bound of failure probability)
    :param generate_data: A name of data generating function
    :param bandit_alg: A name of bandit algorithm
    :param num_experiments: Number of experiments
    :param size_minmax: Minimum and Maximum of signal vector size
    :param seed: A random seed
    :param with_replacement: Whether to sample with replacement
    :param is_normalize: Whether to normalize atoms and signal
    """
    rng = np.random.default_rng(seed)
    sizes_signal = np.linspace(size_minmax[0], size_minmax[1], num_experiments).astype(
        np.int64
    )
    budgets = []
    if generate_data == ADVERSARIAL_CUSTOM:
        generate_data = generate_adversarial_data
        maxmin = (1, 0)
    elif generate_data == NORMAL_CUSTOM:
        generate_data = generate_normal_data
        maxmin = (2, -2)
    else:
        raise NotImplementedError(f"{generate_data} is not implemented")

    atoms, signal = generate_data(len_signal=sizes_signal[-1], seed=seed)
    for size in sizes_signal:
        subset_indices = rng.choice(sizes_signal[-1], size, replace=False)
        atoms_subset, signal_subset = atoms[:, subset_indices], signal[subset_indices]
        if is_normalize:
            atoms_norm = np.linalg.norm(atoms_subset, axis=1, keepdims=True)
            signal_norm = np.linalg.norm(signal_subset, keepdims=True)
            atoms_subset /= atoms_norm
            signal_subset /= signal_norm
            epsilon /= atoms_norm.max() * signal_norm
            maxmin = (
                1,
                -1,
            )  # Assume that the information on element-wise product is not given.
        _, budget = mips_bandit(
            atoms=atoms_subset,
            signal=signal_subset,
            maxmin=maxmin,
            bandit_alg=bandit_alg,
            epsilon=epsilon,
            delta=delta,
            num_best_atoms=1,
            seed=seed,
            is_experiment=False,
            with_replacement=with_replacement,
        )
        budgets.append(budget)
    budgets = np.array(budgets)
    functions = [np.array, np.sqrt, np.log]
    function_names = ["Linear", "Sqrt", "Logarithm"]

    fit_and_plot(sizes_signal, budgets, functions, function_names)


if __name__ == "__main__":
    epsilons = (0.05, 0.1, 0.2, 0.3)
    deltas = (
        0.05,
        0.1,
        0.2,
        0.3,
    )
    # fixed_confidence_exp(
    #     epsilons=epsilons,
    #     deltas=deltas,
    #     bandit_alg=ACTION_ELIMINATION,
    #     num_experiments=50,
    #     with_replacement=True,
    # )
    scaling_exp(
        0.05,
        0.1,
        size_minmax=(10, 10 ** 5),
        num_experiments=10,
        generate_data=NORMAL_CUSTOM,
        bandit_alg=ACTION_ELIMINATION,
        with_replacement=False,
        is_normalize=False,
    )
    scaling_exp(
        0.05,
        0.1,
        size_minmax=(10, 10 ** 5),
        num_experiments=10,
        generate_data=NORMAL_CUSTOM,
        bandit_alg=ACTION_ELIMINATION,
        with_replacement=True,
        is_normalize=False,
    )
    scaling_exp(
        0.05,
        0.1,
        size_minmax=(10, 10 ** 5),
        num_experiments=10,
        generate_data=NORMAL_CUSTOM,
        bandit_alg=MEDIAN_ELIMINATION,
        is_normalize=False,
    )
    # fixed_confidence_exp(
    #     epsilons=epsilons,
    #     deltas=deltas,
    #     bandit_alg=ACTION_ELIMINATION,
    #     num_experiments=10,
    #     with_replacement=False,
    # )
