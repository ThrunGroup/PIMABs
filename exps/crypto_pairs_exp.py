import numpy as np
import pandas as pd
import os
from data.get_data import get_data
from algorithms.mips_bandit import mips_bandit
from utils.constants import (
    BATCH_SIZE,

    # datasets
    NORMAL_CUSTOM,
    ADVERSARIAL_CUSTOM,
    NETFLIX,
    MOVIE_LENS,
    CRYPTO_PAIRS,

    # algorithms
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    GREEDY_MIPS,
    LSH_MIPS,
    PCA_MIPS,
    ASYMMETRIC,

    # scaling specific constants
    TRADEOFF_NUM_EXPERIMENTS,
    TRADEOFF_NUM_ATOMS,
    TRADEOFF_NUM_SIGNALS,
    TRADEOFF_PCA_NUM_EXPERIMENTS,
    TRADEOFF_LSH_NUM_EXPERIMENTS,
    TRADEOFF_DIMENSION,
    NUM_ATOMS_REAL,
    NUM_ATOMS_SIGNAL,
    CRYPTO_PAIRS_SIGNAL_LENGTH,
)


def get_accuracy_array(
        true_candidates: np.ndarray,
        mips_candidates: np.ndarray,
        num_signals,
        num_best_atoms
) -> np.ndarray:
    """
    Returns the average accuracy of the candidates across the signals.
    Note that this is equivalent to recall when the topk atoms is fixed.

    :param true_candidates: the candidates retrieved naively
    :param mip_candidates: the candidates returned by the mips algorithm
    :param returns: an array of accuracies
    """
    accuracy = []
    for idx in range(TRADEOFF_NUM_SIGNALS):
        accuracy.append(len(
            np.intersect1d(true_candidates[idx], mips_candidates[idx])
        ))
    return np.array(accuracy) / (num_signals * num_best_atoms)


def run_exp():
    deltas = [0.001, 0.01, 0.1]
    minmaxs = [(0, 10 ** 4), (0, 10 ** 5), (0, 10 ** 6)]
    num_best_atoms = 1
    num_signals = 3
    num_experiments = len(deltas) * len(minmaxs)
    num_seeds = 5
    num_atoms = 100
    len_signal = 10 ** 6

    avg_accuracy = np.zeros(num_experiments)
    avg_accuracy_std = np.zeros(num_experiments)

    for algorithm in [ACTION_ELIMINATION, ADAPTIVE_ACTION_ELIMINATION]:
        print("Running ", algorithm)
        for seed in range(num_seeds):
            print("Seed: ", seed)

            atoms, signals = get_data(
                num_atoms=num_atoms,
                len_signal=len_signal,
                num_signals=num_signals,
                seed=seed,
                data_type=CRYPTO_PAIRS,
            )

            # these are the oracle MIPS atoms
            naive_candidates_array = (
                np.matmul(-atoms, signals.transpose())
                    .argsort(axis=0)[:num_best_atoms]
                    .transpose()
            )

            accuracies = []
            accuracies_std = []
            for delta in deltas:
                for minmax in minmaxs:
                    candidates_array, budget_array = mips_bandit(
                        atoms=atoms,
                        signals=signals,
                        maxmin=minmax,
                        bandit_alg=algorithm,
                        epsilon=0.0,
                        delta=delta,
                        num_best_atoms=num_best_atoms,
                        seed=seed,
                        is_experiment=False,
                        with_replacement=False,
                        batch_size=4,
                    )

                    accuracy_array = get_accuracy_array(naive_candidates_array, candidates_array, num_signals,
                                                        num_best_atoms)
                    accuracies.append(np.sum(accuracy_array))
                    accuracies_std.append(np.std(accuracy_array) / np.sqrt(num_signals))

            avg_accuracy += np.array(accuracies)
            avg_accuracy_std += np.array(accuracies_std)

        avg_accuracy /= num_seeds
        avg_accuracy_std /= num_seeds

        log_dict = {
            "data_type": 'crypto_pairs',
            "accuracy": avg_accuracy,
            "accuracy_std": avg_accuracy_std,
        }
        log_df = pd.DataFrame(log_dict)
        current_file_directory = os.path.dirname(__file__)
        log_dir = os.path.join(current_file_directory, "logs/crypto_pairs")
        filename = os.path.join(
            log_dir,
            algorithm
            + "_"
              "SPEEDUP_"
            + log_dict["data_type"]
            + "_"
            + "atoms"
            + str(num_atoms)
            + "signal_size"
            + str(len_signal)
            + "_"
            + "topk"
            + str(num_best_atoms)
            + ".csv",
        )
        os.makedirs(log_dir, exist_ok=True)
        log_df.to_csv(filename, index=False)


if __name__ == "__main__":
    run_exp()
