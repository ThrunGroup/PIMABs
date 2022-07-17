import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from typing import Callable, Tuple
from constants import ADVERSARIAL_CUSTOM, MEDIAN_ELIMINATION, SUCCESSIVE_ELIMINATION
from utils import welford_variance_calc, get_wo_variance


def mips_bandit(
        items_data: np.ndarray,
        query: np.ndarray,
        maxmin: Tuple,
        bandit_alg: str,
        epsilon: float,
        delta: float,
        num_best_items: int = 1,
        seed: int = 0,
        is_experiment: bool = False,
        verbose: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Solve MIPS(Maximum Inner Product Search) problem using a multi-armed bandit algorithm with fixed confidence setting.
    MIPS problem: To find a best item vector in data_items set that has maximum inner product with a query vector.

    @param items_data: 2d array that contains items in rows. Items are the vectors one of which will be selected
    as the vector with maximum inner product with a query vector.
    @param query: A query vector.
    @param maxmin: A tuple with two elements which are the maximum and minimum of data_items array
    @param bandit_alg: A name of multi-armed bandit algorithm
    @param epsilon: A maximum additive error between the true best arm and the best arm identified by our
    bandit algorithm
    @param delta: A probability that the difference between the reward of the true best arm and the identified
    best arm is beyond epsilon
    @param num_best_items: Number of items that bandit algorithm identify as top/best arms.
    @param seed: A random seed
    @param is_experiment: Whether to do worst-case experiment
    @param verbose: Verbosity of function
    @return: Indices of best item that have maximum inner product with a query vector and the budget (total number of
    calculations)
    """
    assert len(items_data.shape) == 2, "Invalid size of data items matrix"
    assert len(query.shape) == 1, "Invalid size of query vector"
    rng = np.random.default_rng(seed)

    if bandit_alg is MEDIAN_ELIMINATION:
        # Refer to Algorithm 1 in "shorturl.at/adPR3"
        len_query = query.shape[0]
        candidates = np.arange(items_data.shape[0])
        estimates = np.zeros(items_data.shape[0])
        population_idcs = np.arange(len_query)
        new_epsilon = epsilon / 4
        new_delta = delta / 2
        prev_T = 0  # A helper variable for calculating the optimal number of samples in each round
        total_num_samples = 0
        budget = 0  # Total number of calculations

        # U = (math.log(1 / delta) * (maxmin[0] - maxmin[1]) ** 2) / (2 * epsilon ** 2)
        def m(u):
            """
            A helper function for calculating the optimal number of samples
            """
            return min(
                (u + 1) / (1 + u / len_query), (u + u / len_query) / (1 + u / len_query)
            )

        while len(candidates) > num_best_items:
            # Calculate the optimal number of samples in the current round
            new_T = m(
                2
                * math.log(
                    2
                    * (len(candidates) - num_best_items)
                    / (
                            new_delta
                            * (math.floor(((len(candidates)) - num_best_items) / 2) + 1)
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
                    len_query - total_num_samples, current_num_samples, replace=False,
                )
            else:
                # With an adversarial dataset, wan do a worst-case experiment by not shuffling a data
                # Ex) items_data = np.transpose(np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                #                                         [1, 1, 1, 1, 0, 0, 0, 0]])
                #     query = np.ones(8)
                #
                #     In this case, sampling without shuffling will cause a poor performance since
                #     a first few elements of each item doesn't represent their distribution well.
                #     So, it's a worst case scenario.
                idcs = np.arange(current_num_samples, dtype=np.int64)
            sample_idcs = population_idcs[idcs]
            population_idcs = np.delete(population_idcs, idcs)

            estimates[candidates] = (
                                            estimates[candidates] * total_num_samples
                                            + (
                                                np.matmul(
                                                    items_data[:, sample_idcs][candidates], query[sample_idcs]
                                                )
                                            )
                                    ) / (total_num_samples + current_num_samples)

            candidates = candidates[
                np.argsort(estimates[candidates])
            ]  # sort candidates with estimates
            candidates = candidates[math.ceil((len(candidates) - num_best_items) / 2):]
            new_epsilon *= 3 / 4
            new_delta *= 1 / 2
            total_num_samples += current_num_samples
        if verbose:
            print(f"budget is {budget}")
        return candidates, budget
    elif bandit_alg is SUCCESSIVE_ELIMINATION:
        num_items = items_data.shape[0]
        len_query = items_data.shape[1]
        candidates = np.arange(num_items)
        lcbs = -np.inf * np.ones(num_items)  # lower confidence bounds of arm returns
        ucbs = np.inf * np.ones(num_items)  # upper confidence bounds of arm returns
        estimates = np.zeros(num_items)  # estimated means of arm returns
        variances = np.zeros(num_items)  # variance of the mean of arm returns
        deltas = np.zeros(
            num_items
        )  # the half-widths of the confidence intervals (ucb = mean + delta, lcb = mean - delta)
        population_idcs = np.arange(num_items)  # Indices of items population that are not sampled
        batch_size = 30  # Hard-coded
        t = 0  # Time step
        while len(candidates) > num_best_items:
            if not is_experiment:
                idcs = rng.choice(
                    len_query - t * batch_size, batch_size, replace=False,
                )
            else:
                # With an adversarial dataset, wan do a worst-case experiment by not shuffling a data
                # Ex) items_data = np.transpose(np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                #                                         [1, 1, 1, 1, 0, 0, 0, 0]])
                #     query = np.ones(8)
                #
                #     In this case, sampling without shuffling will cause a poor performance since
                #     a first few elements of each item doesn't represent their distribution well.
                #     So, it's a worst case scenario.
                idcs = np.arange(batch_size, dtype=np.int64)
            sample_idcs = population_idcs[idcs]
            population_idcs = np.delete(population_idcs, idcs)
            sampled_returns = items_data[:, sample_idcs][candidates] * np.expand_dims(query[sample_idcs], 0)

            estimates = (estimates * batch_size * t + np.mean(sampled_returns, axis=1)) / ((t + 1) * batch_size)
            ##### More to implement





    else:
        raise NotImplementedError(f"{bandit_alg} is not implemented.")


def generate_adversarial_data(
        num_items: int = 10 ** 3, len_query: int = 10 ** 4, seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an adversarial dataset with given parameters. Refer to experiment (1) in "shorturl.at/adPR3".

    @param num_items: Number of items(arms)
    @param len_query: A size of query vector
    @param seed: A random seed
    @return: An adversarial dataset for MIPS problem. Returns items(2d-array) and a query vector(1d-array).
    """
    rng = np.random.default_rng(seed)
    items_array = np.zeros((num_items, len_query))
    p = rng.random(num_items)
    for idx in range(num_items):
        items_array[idx][: math.floor(p[idx] * len_query)] = 1
    query = np.ones(len_query)
    return items_array, query


def fixed_confidence_exp(
        epsilons: Tuple[float],
        deltas: Tuple[float],
        generate_data: str = ADVERSARIAL_CUSTOM,
        bandit_alg: str = MEDIAN_ELIMINATION,
        num_experiments: int = 20,
        verbose: bool = False,
) -> None:
    """
    Validate that bandit algorithm for fixed confidence setting works well. In other words, observed failure
    probability for bandit algorithm with fixed confidence setting should be lower than failure probability(delta)
    passed as a parameter for a bandit algorithm.

    @param epsilons: A tuple of epsilons
    @param deltas: A tuple of deltas
    @param generate_data: A name of data generating function
    @param bandit_alg: A name of bandit algorithm
    @param num_experiments: Number of experiments for each fixed confidence setting.
    @param verbose: verbosity of function
    """
    if generate_data == ADVERSARIAL_CUSTOM:
        generate_data = generate_adversarial_data
        maxmin = (1, 0)
    else:
        raise NotImplementedError(f"{generate_data} is not implemented")
    for epsilon in epsilons:
        for delta in deltas:
            failure_counts = 0
            print(f"Doing an experiment with epsilon={epsilon} and delta={delta}")
            for seed in range(num_experiments):
                if verbose:
                    print(f"-seed: {seed}")
                items_data, query = generate_data(seed=seed)
                if verbose:
                    print("--data generated")
                true_inner_products = np.matmul(items_data, query)
                true_best_idx = np.argmax(true_inner_products)
                if verbose:
                    print("--naively solved")
                estimated_best_idx, _ = mips_bandit(
                    items_data=items_data,
                    query=query,
                    maxmin=maxmin,
                    bandit_alg=bandit_alg,
                    epsilon=epsilon,
                    delta=delta,
                    num_best_items=1,
                    seed=seed,
                    is_experiment=False,
                )
                if verbose:
                    print("--stochastically solved")
                best_arm_mean = true_inner_products[true_best_idx] / query.shape[0]
                estimated_best_arm_mean = (
                        true_inner_products[estimated_best_idx][0] / query.shape[0]
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
) -> None:
    """
    Check that the time complexity of bandit algorithm scales with the square root of the query size.
    MEDIAN_ELIMINATION algorithm should satisfy this scaling according to Corollary 3 in "shorturl.at/adPR3".

    @param epsilon: A epsilon (an additive error allowed)
    @param delta: A delta (an upper bound of failure probability)
    @param generate_data: A name of data generating function
    @param bandit_alg: A name of bandit algorithm
    @param num_experiments: Number of experiments
    @param size_minmax: Minimum and Maximum of query vector size
    @param seed: A random seed
    """
    rng = np.random.default_rng(seed)
    sizes_query = np.linspace(size_minmax[0], size_minmax[1], num_experiments).astype(np.int64)
    budgets = []
    if generate_data == ADVERSARIAL_CUSTOM:
        generate_data = generate_adversarial_data
        maxmin = (1, 0)
    else:
        raise NotImplementedError(f"{generate_data} is not implemented")

    for size in sizes_query:
        items_data, query = generate_data(len_query=size, seed=rng.integers(low=1000000, size=1))
        _, budget = mips_bandit(
            items_data=items_data,
            query=query,
            maxmin=maxmin,
            bandit_alg=bandit_alg,
            epsilon=epsilon,
            delta=delta,
            num_best_items=1,
            seed=seed,
            is_experiment=True,
        )
        budgets.append(budget)
    budgets = np.array(budgets)
    mean_budget = np.mean(budgets)

    lr = LinearRegression()
    lr.fit(sizes_query.reshape(-1, 1), budgets)

    sqrt_lr = LinearRegression()
    sqrt_lr.fit(np.sqrt(sizes_query).reshape(-1, 1), budgets)

    hashes = np.linspace(size_minmax[0], size_minmax[1], 5000)
    hashes_predict = lr.predict(hashes.reshape(-1, 1))
    hashes_predict_sqrt = sqrt_lr.predict(np.sqrt(hashes.reshape(-1, 1)))

    budgets_predict = lr.predict(sizes_query.reshape(-1, 1))
    budgets_predict_sqrt = sqrt_lr.predict(np.sqrt(sizes_query).reshape(-1, 1))
    hashes_R2 = 1 - (
            ((budgets_predict.reshape(-1) - budgets) ** 2).sum()
            / ((mean_budget - budgets) ** 2).sum()
    )
    hashes_log_R2 = 1 - (
            ((budgets_predict_sqrt.reshape(-1) - budgets) ** 2).sum()
            / ((mean_budget - budgets) ** 2).sum()
    )

    plt.scatter(sizes_query, budgets)
    plt.plot(hashes, hashes_predict, color="green", label=f"Linear fit, $R^2 = {hashes_R2}$")
    plt.plot(
        hashes,
        hashes_predict_sqrt,
        color="orange",
        label=f"Sqrt fit, $R^2 = {hashes_log_R2}$",
    )
    plt.ylabel("Number of calculations for MIPS")
    plt.xlabel("Query vector size")
    plt.legend(loc="upper left")
    plt.show()

    assert hashes_R2 < hashes_log_R2, "Number of calculations are b"


if __name__ == "__main__":
    epsilons = (0.99, 0.4, 0.05, 0.1, 0.2)
    deltas = (0.9999, 0.05, 0.1, 0.2, 0.3,)
    fixed_confidence_exp(epsilons, deltas, num_experiments=50)
    scaling_exp(0.05, 0.1, size_minmax=(10, 10 ** 5), num_experiments=50)
