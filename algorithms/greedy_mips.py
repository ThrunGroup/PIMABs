import numba as nb
import numpy as np
import heapq
import time
from numba.typed import List

from utils.utils import fit_and_plot, r


def generate_conditer(atoms: np.ndarray) -> np.ndarray:
    return np.argsort(-atoms, axis=0)


@nb.njit
def generate_heap(atoms: np.ndarray, conditer: np.ndarray, signal: np.ndarray):
    """
    Query-dependent pre-processing procedure in Greedy-MIPS
    """
    for t in range(signal.shape[0]):
        if signal[t] > 0:
            z = -atoms[conditer[0, t], t] * signal[t]  # To make max-heapq
        else:
            z = -atoms[conditer[-1, t], t] * signal[t]
        if t == 0:
            Q = [(z, t)]
        else:
            heapq.heappush(Q, (z, t))
    return Q, np.zeros(signal.shape[0], dtype=np.int32)


@nb.njit
def candidate_screening(
    atoms: np.ndarray, signal:np.ndarray, conditer:np.ndarray, iters:np.ndarray, Q:List, budget: int, visited_j: np.ndarray
):
    """
    Screening candidates that have maximum one-element inner product with signal
    """
    candidates = []
    while len(candidates) < budget:
        z, t = heapq.heappop(Q)
        z = -z  # heapq is min heap but regard it as max heap
        conditer_idx = iters[t]
        j = (
            conditer[conditer_idx, t]
            if signal[t] > 0
            else conditer[-1 - conditer_idx, t]
        )
        if visited_j[j] == 0:
            candidates.append(j)
            visited_j[j] = 1
        while conditer_idx < conditer.shape[0]:
            iters[t] += 1
            conditer_idx = iters[t]
            j = (
                conditer[conditer_idx, t]
                if signal[t] > 0
                else conditer[-1 - conditer_idx, t]
            )
            if visited_j[j] == 0:
                z = atoms[j, t] * signal[t]
                heapq.heappush(Q, (-z, t))
                break
    return np.array(candidates)


def greedy_mips(
    atoms:np.ndarray, signals:np.ndarray, budget: int =10, verbose: bool =False, num_best_atoms:int =1, conditer:np.ndarray=None,
):
    """
    Run Greedy MIPS
    - Description: Find atoms(the number of atoms is set by budget param) that have maximum one-element product
    with signal. With these atoms, find the best atom(s) by naively computing inner product.
    """

    start_time = time.time()
    if conditer is None:
        conditer = generate_conditer(atoms)
    visited_j = np.zeros(atoms.shape[0])
    if verbose:
        print(f"preprocess time: {time.time() - start_time}")
    best_atom_list = []
    query_dependent_time_list = []
    candidate_ranking_time_list = []
    for signal in signals:
        start_time = time.time()
        heap, iters = generate_heap(atoms, conditer, signal)
        heap = List(heap)
        candidates = candidate_screening(
            atoms, signal, conditer, iters, heap, budget, visited_j
        )
        # Uncomment following line to check if algorithm is correctly implemented
        # print(np.sum(candidates == np.argsort(-np.max(atoms * np.expand_dims(sigIal, axis=0), axis=1))[:budget])/ budget)
        query_dependent_time = time.time() - start_time
        start_time = time.time()
        best_atom_list.append(
            candidates[np.argsort(-atoms[candidates] @ signal)[:num_best_atoms]]
        )
        candidate_ranking_time = time.time() - start_time
        query_dependent_time_list.append(query_dependent_time)
        candidate_ranking_time_list.append(candidate_ranking_time)
        if verbose:
            print(
                f"query dependent preprocess + candidates screening: {query_dependent_time}"
            )
            print(f"candidate ranking: {candidate_ranking_time}")
        for candidate in candidates:
            visited_j[candidate] = 0
    return best_atom_list, query_dependent_time_list, candidate_ranking_time_list


def reproduce_figure6():
    num_experiments = 50
    num_best_atoms = 5
    rng = np.random.default_rng(0)
    for n in [17, 18, 19]:
        for k in [2, 5, 7]:
            speedup_list = []
            accuracy_list = []
            atoms = rng.normal(size=(2 ** n, 2 ** k))
            signals = rng.normal(size=(num_experiments, 2 ** k))
            time_list = []
            naive_best_arms = []
            for signal in signals:
                start_time = time.time()
                naive_best_arms.append(np.argsort(-atoms @ signal)[:num_best_atoms])
                time_list.append(time.time() - start_time)
            time_list = np.array(time_list)
            for budget in np.geomspace(2, 500, 20):
                conditer = generate_conditer(atoms)
                best_arms, query_dependent_time_list, ranking_time_list = greedy_mips(
                    atoms,
                    signals,
                    int(2 ** n / budget),
                    num_best_atoms=num_best_atoms,
                    conditer=conditer,
                )
                best_arms = np.array(best_arms)
                accuracy = 0
                for idx in range(len(best_arms)):
                    accuracy += len(
                        np.intersect1d(naive_best_arms[idx], best_arms[idx])
                    )
                accuracy /= len(best_arms) * len(best_arms[0])

                speedup_list.append(
                    np.mean(time_list)
                    / (
                        np.mean(query_dependent_time_list[1:])
                        + np.mean(ranking_time_list[1:])
                    )
                )
                accuracy_list.append(accuracy)
            fit_and_plot(
                np.array(speedup_list),
                np.array(accuracy_list),
                [np.array, np.sqrt, np.log],
                ["Linear", "Sqrt", "Log"],
                title=f"Greedy MIPS: n = 2 ^ {n}, k = 2 ^ {k}",
                xlabel=f"Speed up over naive approaches({r(np.mean(time_list))}",
                ylabel=f"Performance (prec @ {num_best_atoms})",
            )


if __name__ == "__main__":

    reproduce_figure6()
    # from data.custom_data import generate_custom_data
    #
    # rng = np.random.default_rng(3)
    # atoms = rng.normal(size=(2 ** 18, 20))
    # signals = rng.normal(size=(50, 20))
    # naive_best_arms = []
    # times_list = []
    # for signal in signals:
    #     start_time = time.time()
    #     naive_best_arms.append(np.argsort(-atoms @ signal)[:5])
    #     times_list.append(time.time() - start_time)
    # # print(f"best arm: {naive_best_arms}")
    # # print(np.argsort(-np.max(atoms * np.expand_dims(signal, axis=0), axis=1))[:10])
    # print("naive approach:", np.mean(times_list))
    #
    # best_arms, query_dependent_time_list, ranking_time_list = greedy_mips(
    #     atoms, signals, 2 ** 18 / 30, num_best_atoms=5
    # )
    # print(np.mean(query_dependent_time_list[1:]), np.mean(ranking_time_list[1:]))
    # # print(f"best arm by greedy mips: {best_arms}")
    # accuracy = 0
    # for idx in range(len(best_arms)):
    #     accuracy += len(np.intersect1d(naive_best_arms[idx], best_arms[idx]))
    # accuracy /= len(best_arms) * len(best_arms[0])
    # print(accuracy)
