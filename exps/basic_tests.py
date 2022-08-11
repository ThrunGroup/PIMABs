import numpy as np
import math
from typing import Tuple

from algorithms.lsh_mips import LSH_MIPS
from algorithms.pca_mips import PCA_MIPS

from utils.constants import (
    ADVERSARIAL_CUSTOM,
    NORMAL_CUSTOM,
    NORMAL_PAPER,
    PCA,
    LSH,
    SYMMETRIC,
    ASYMMETRIC,
)
from utils.utils import subset_2d, fit_and_plot
from data.custom_data import generate_normal_data, generate_adversarial_data, generate_normal_paper


def run_pca_mips(data, query, top_k=[1, 3], delta=1, verbose=False):
    """
    Finds the top_k candidates to the MIPS problem by building the pca-tree data structure. Returns an array of
    budgets and candidates for each of the top k values.

    :delta: the number of top principal components to use
    :top_k: array of top k candidates we want to find
    """
    if verbose:
        print("... running pca-mips")
    PCAMIPS = PCA_MIPS(data, query, delta=delta)
    PCAMIPS.preprocess()

    budgets = []
    results = []
    for k in top_k:
        budget, candidates = PCAMIPS.run(top_k=k)
        budgets.append(budget)
        results.append(candidates)

        if verbose:
            print(f" =>> Top {top_k} candidates: ", candidates)
            print(" =>> budget: ", budget)

    return budgets, results


def run_lsh_mips(
    data,
    query,
    top_k=[1, 3],
    num_bits=5,
    num_hfunc=5,
    num_tables=10,
    type=ASYMMETRIC,
    verbose=False
):
    """
    Finds the top_k candidates to the MIPS problem by building the LSH hash table. Returns an array of
    budgets and candidates for each of the top k values.

    :num_hfunc: the number of hash functions per table (i.e. size of table). The final hash value is an
                AND-construction of the hash functions
    :num_tables: the number of hash tables. The final result is an OR-construction of the hash values
                 coming from each table.
    :type: either asymmetric or symmetric. Dictates which hash function to use.
    """
    if verbose:
        print(f"... Running LSH-MIPS {type}")
    LSHMIPS = LSH_MIPS(
        data,
        query,
        num_bits=num_bits,
        num_hfunc=num_hfunc,
        num_tables=num_tables,
        type=type
    )
    LSHMIPS.prepare_data()
    LSHMIPS.create_lsh_table(LSHMIPS.ext_data)

    budgets = []
    results = []
    for k in top_k:
        budget, candidates = LSHMIPS.run(
            LSHMIPS.ext_data,
            LSHMIPS.ext_query,
            top_k=k
        )
        budgets.append(budget)
        results.append(candidates)

        if verbose:
            print(f" =>> Top {top_k} candidate(s): ", candidates)
            print(" =>> budget: ", budget)

    return budgets, results


def naive_mips(data, query, top_k):
    """
    Returns the solution to MIPS using the naive search. The solution is an array of indices of the
    top k (default 1) rows in the dataset. Returns an array of budgets and candidates for each of the top k values.

    :top_k: a list of the number of closest vectors we want to return
    """
    assert (data.shape[1] == len(query))

    result = []
    dist_vec = np.matmul(data, query)
    for k in top_k:
        idx = np.argpartition(-dist_vec, k - 1)  # -dist_vec to we can get the top k largest
        result.append(idx[:k])
    return result


def compare_with_naive(candidates1, candidates2, top_k):
    """
    Compares the top k candidates from candidates1 and candidates2 and returns the precision. The precision is
    defined as the fraction of true top K solutions in the returned top K solutions

    :candidates1: a list of true candidates returned from the naive search for each k value
    :candidates2: a list of candidates returned from either LSH-MIPS or PCA-MIPS.
    """
    if len(candidates2) == 0:
        return 0

    for i, k in enumerate(top_k):
        precision = len(np.intersect1d(candidates1, candidates2)) / k
        #print("     => precision: ", precision)
        return precision


def scaling_baseline(
    top_k: list[int] = [5],
    generate_data: str = ADVERSARIAL_CUSTOM,
    name_alg: str = PCA,
    size_minmax: Tuple[int] = (10, 10 ** 4),
    type: str = None,
    num_bits: int = 5,
    num_hfunc: int = 10,
    num_tables: int = 5,
    num_experiments: int = 20,
    seed: int = 0,
    is_normalize: bool = True,
) -> None:
    """
    Verify that the query time complexities of pca-mips and lsh-mips are both linear to the size of the
    signal vector. The exact time complexities are as follows:
        pca-mips: O(n * N / 2^d).
        lsh-mips: O(n * N * b / 2^a)

    :top_k: A list of how many candidates to return
    :generate_data: A name of data generating function
    :name_alg: A name of algorithm (either pca or lsh)
    :size_minmax: the minimum, maximum of the signal vector size

    :type: either symmetric of asymmetric
    :num_bits: number of buckets to hash the dataset to
    :num_hfunc: number of hyperplanes per hash table
    :num_tables: number of hash tables

    :num_experiments: Number of experiments
    :seed: A random seed
    :is_normalize: Whether to normalize atoms and signal
    """
    rng = np.random.default_rng(seed)
    sizes_signal = np.linspace(size_minmax[0], size_minmax[1], num_experiments).astype(
        np.int64
    )
    budgets = []
    #print(f"...generating {generate_data}")
    if generate_data == ADVERSARIAL_CUSTOM:
        generate_data = generate_adversarial_data
    elif generate_data == NORMAL_CUSTOM:
        generate_data = generate_normal_data
    elif generate_data == NORMAL_PAPER:
        generate_data = generate_normal_paper
    else:
        raise NotImplementedError(f"{generate_data} is not implemented")

    atoms, signal = generate_data(num_atoms=1000, len_signal=sizes_signal[-1], seed=seed)
    for size in sizes_signal:
        subset_indices = rng.choice(sizes_signal[-1], size, replace=False)
        atoms_subset, signal_subset = atoms[:, subset_indices], signal[subset_indices]
        if is_normalize:
            atoms_norm = np.linalg.norm(atoms_subset, axis=1, keepdims=True)
            signal_norm = np.linalg.norm(signal_subset, keepdims=True)
            atoms_subset /= atoms_norm
            signal_subset /= signal_norm

        #print("=> signal length:", size)
        if name_alg == PCA:
            budget, _ = run_pca_mips(
                data=atoms_subset,
                query=signal_subset,
                delta=2,
                top_k=top_k
            )
        elif name_alg == LSH:
            #print(f"=> type is {type}")
            budget, _ = run_lsh_mips(
                type=type,
                data=atoms_subset,
                query=signal_subset,
                num_bits=num_bits,
                num_hfunc=num_hfunc,
                num_tables=num_tables,
                top_k=top_k,
            )

        budgets.append(budget)
    budgets = np.array(budgets)
    return budgets, sizes_signal


def precision_speed(
    top_k: list[int] = [5],
    generate_data: str = ADVERSARIAL_CUSTOM,
    name_alg: str = PCA,
    a_minmax: Tuple[int] = (0, 9),
    type: str = None,
    num_experiments: int = 4,
    seed: int = 0,
    is_normalize: bool = True,
    num_hfunc: int = 10,
    num_tables: int = 5,
) -> None:
    """
    Verify the precision, speed tradeoff of pca-mips and lsh-mips is logarithmic.
    Precisions is defined as the fraction of true top K solutions in the returned tp K solutions and
    speed is dictated by the total budget used.

    :top_k: A list of how many candidates to return
    :generate_data: A name of data generating function
    :name_alg: A name of algorithm (either pca or lsh)
    :a_minmax: the min and max of "a" where a represents either the depth of pca-tree (pca-mips) or the number of bits
                to hash the dataset into (lsh-mips)
    :type: either symmetric or asymmetric
    :num_experiments: Number of experiments
    :seed: A random seed
    :is_normalize: Whether to normalize atoms and signal
    :num_hfunc: number of hyperplanes per hash table
    :num_tables: number of hash tables
    """
    rng = np.random.default_rng(seed)
    a_values = np.linspace(a_minmax[0], a_minmax[1], num_experiments).astype(
        np.int64
    )

    budgets = []
    #print(f"... generating {generate_data}")
    if generate_data == ADVERSARIAL_CUSTOM:
        generate_data = generate_adversarial_data
    elif generate_data == NORMAL_CUSTOM:
        generate_data = generate_normal_data
    elif generate_data == NORMAL_PAPER:
        generate_data = generate_normal_paper
    else:
        raise NotImplementedError(f"{generate_data} is not implemented")

    atoms, signal = generate_data(num_atoms=1000, len_signal=10000, seed=seed)
    if is_normalize:
        atoms_norm = np.linalg.norm(atoms, axis=1, keepdims=True)
        signal_norm = np.linalg.norm(signal, keepdims=True)
        atoms /= atoms_norm
        signal /= signal_norm

    budgets = []
    precisions = []
    candidates = naive_mips(atoms, signal, top_k=top_k)
    for a in a_values:
        #print("=> a: ", a)
        if name_alg == PCA:
            budget, _candidates = run_pca_mips(
                data=atoms,
                query=signal,
                delta=a,  # this deepens the tree
                top_k=top_k
            )
        elif name_alg == LSH:
            budget, _candidates = run_lsh_mips(
                type=type,
                data=atoms,
                num_bits=a,  # this controls the final buckets
                num_hfunc=num_hfunc,
                num_tables=num_tables,
                query=signal,
                top_k=top_k,
            )
        budgets.append(budget)
        precisions.append(compare_with_naive(candidates, _candidates, top_k))
    return np.array(budgets), np.array(precisions).reshape((len(precisions), 1))



