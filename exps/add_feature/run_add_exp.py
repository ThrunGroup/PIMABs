import numpy as np
from tqdm import tqdm

from data.get_data import get_data
from utils.constants import NETFLIX, ACTION_ELIMINATION
from algorithms.bandit_mips_add import BanditMIPSAdd


def are_same_arrays(A, B):
    """
    Check if two arrays share the same elements

    :param A: Numpy array to compare (best_atoms_from_entire_dataset)
    :param B: Numpy array to compare (best_atoms_after_addition)
    :return: Whether they share the same elements
    """
    a_flat = np.sort(A, axis=0).flatten()
    b_flat = np.sort(B, axis=0).flatten()
    return np.array_equal(a_flat, b_flat)


def add_exp(seed: int = 0):
    """
    An experiment to show that BanditMIPS supports a correct addition feature

    Run BanditMIPS on two conditions
    1. Run it on the entire Netflix dataset
    2. Run it on the first half, get its top_k, and then run it on the second half

    :param seed: Seed for getting data from the Netflix dataset
    :return: Whether the best atoms retrieved from both conditions should match
    """
    # Get Netflix dataset
    num_atoms = 1000
    atoms, signals = get_data(
        num_atoms=num_atoms,
        len_signal=10000,
        num_signals=1,
        seed=seed,
        data_type=NETFLIX,
    )

    epsilon, delta, maxmin = 0.0001, 0.001, (2, 0)

    bandit_mips = BanditMIPSAdd()

    # Run BanditMIPS on the entire dataset
    best_atoms_from_entire_dataset = bandit_mips.search_top_k(
        atoms=atoms,
        signals=signals,
        maxmin=maxmin,
        bandit_alg=ACTION_ELIMINATION,
        epsilon=epsilon,
        delta=delta,
        num_best_atoms=1,
        save_top_k=False,
    )

    # Run BanditMIPS on the first half
    bandit_mips.search_top_k(
        atoms=atoms[:num_atoms // 2],
        signals=signals,
        maxmin=maxmin,
        bandit_alg=ACTION_ELIMINATION,
        epsilon=epsilon,
        delta=delta,
        num_best_atoms=1,
    )

    # Run BanditMIPS on the second half
    best_atoms_after_addition = bandit_mips.search_top_k(
        atoms=atoms[num_atoms // 2:],
        signals=signals,
        maxmin=maxmin,
        bandit_alg=ACTION_ELIMINATION,
        epsilon=epsilon,
        delta=delta,
        num_best_atoms=1,
        is_addition=True,  # Let the algorithm know that this a new data stream
    )

    return are_same_arrays(best_atoms_from_entire_dataset, best_atoms_after_addition)


def run_exp(num_experiments: int = 100):
    """
    Run the addition experiment multiple times

    :param num_experiments: Number of experiments to run
    """
    exp_results = []
    for seed in tqdm(range(num_experiments)):
        exp_results += add_exp(seed),

    num_correct = np.sum(exp_results)
    print("Correct experiments (%): ", num_correct / num_correct * 100)


if __name__ == "__main__":
    run_exp()
