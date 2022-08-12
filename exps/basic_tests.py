import numpy as np
import math
from typing import Tuple, List

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
    NUM_HFUNC,
    NUM_POINTS_LSH
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
    precisions = []
    for i, k in enumerate(top_k):
        if len(candidates2[i]) == 0:
            precisions.append(0.0)
        else:
            precisions.append(len(np.intersect1d(candidates1[i], candidates2[i])) / k)
    return precisions


def scaling_baseline(
    seed: int = 0,
    top_k: List[int] = [5],
    generate_data: str = ADVERSARIAL_CUSTOM,
    is_normalize: bool = False,
    size_minmax: Tuple[int] = (10, 10 ** 4),
    num_points: int = 20,
    name_alg: str = PCA,
    num_hfunc: int = NUM_HFUNC//3,
    num_tables: int = 3,
    type: str = None,
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
    :num_hfunc: number of hyperplanes per hash table
    :num_tables: number of hash tables

    :num_experiments: Number of experiments
    :seed: A random seed
    :is_normalize: Whether to normalize atoms and signal
    """
    rng = np.random.default_rng(seed)
    sizes_signal = np.linspace(size_minmax[0], size_minmax[1], num_points).astype(
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

    atoms, signal = generate_data(num_atoms=100, len_signal=sizes_signal[-1], seed=seed)
    for size in sizes_signal:
        subset_indices = rng.choice(sizes_signal[-1], size, replace=False)
        atoms_subset, signal_subset = atoms[:, subset_indices], signal[subset_indices]
        if is_normalize:
            atoms_norm = np.linalg.norm(atoms_subset, axis=1, keepdims=True)
            signal_norm = np.linalg.norm(signal_subset, keepdims=True)
            atoms_subset /= atoms_norm
            signal_subset /= signal_norm

        print("     ... signal length:", size)
        if name_alg == PCA:
            budget, _ = run_pca_mips(
                data=atoms_subset,
                query=signal_subset,
                delta=2,
                top_k=top_k
            )
        elif name_alg == LSH:
            budget, _ = run_lsh_mips(
                type=type,
                data=atoms_subset,
                query=signal_subset,
                num_hfunc=num_hfunc,
                num_tables=num_tables,
                top_k=top_k,
            )

        budgets.append(budget)
    budgets = np.array(budgets)
    return np.squeeze(budgets), np.array(sizes_signal)


def precision_speed_pca(
    seed: int = 0,
    top_k: List[int] = [5, 10],
    generate_data: str = ADVERSARIAL_CUSTOM,
    is_normalize: bool = False,
    num_points: int = 4,
    depth_minmax: Tuple[int] = (0, 5),
) -> None:
    """
    For a given dataset and a certain minmax range, computes the budgets and precisions for
    num_points total experiments of pca-mips. Returns a numpy array of budgets and their corresponding precision.

    :seed: A random seed
    :top_k: A list of how many candidates to return
    :generate_data: A name of data generating function
    :is_normalize: Whether to normalize atoms and signal
    :num_points: Number of points that will be plotted
    :depth_minmax: the min and max of depth of pca-tree
    """
    # generate the correct dataset
    if generate_data == ADVERSARIAL_CUSTOM:
        generate_data = generate_adversarial_data
    elif generate_data == NORMAL_CUSTOM:
        generate_data = generate_normal_data
    elif generate_data == NORMAL_PAPER:
        generate_data = generate_normal_paper
    else:
        raise NotImplementedError(f"{generate_data} is not implemented")
    #atoms, signal = generate_data(num_atoms=1000, len_signal=10000, seed=seed)
    atoms, signal = generate_data(num_atoms=1000, len_signal=10000, seed=seed)

    # by default, we will NOT be needing to normalize
    if is_normalize:
        atoms_norm = np.linalg.norm(atoms, axis=1, keepdims=True)
        signal_norm = np.linalg.norm(signal, keepdims=True)
        atoms /= atoms_norm
        signal /= signal_norm

    # setup experiements by defining the depths we will experiment on.
    # This is analogous to setting "a" in the paper
    depths = np.linspace(depth_minmax[0], depth_minmax[1], num_points).astype(
        np.int64
    )

    # running the experiments and concatenating the results in the budget, precisions list.
    # both lists should be of size num_points at the end of the experiment
    budgets = []
    precisions = []
    candidates = naive_mips(atoms, signal, top_k=top_k)
    for depth in depths:
        print("     ... depth: ", depth)
        budget, _candidates = run_pca_mips(
            data=atoms,
            query=signal,
            delta=depth,  # this deepens the tree
            top_k=top_k
        )
        budgets.append(np.array(budget))
        precisions.append(
            np.array(compare_with_naive(candidates, _candidates, top_k))
        )
    return np.array(budgets), np.array(precisions)


def precision_speed_lsh(
    seed: int = 0,
    top_k: list[int] = [5, 10],
    generate_data: str = ADVERSARIAL_CUSTOM,
    is_normalize: bool = False,
    num_points: int = 10,
    hfunc_minmax: Tuple[int] = (0, NUM_HFUNC),
    table_minmax: Tuple[int] = (1, 6),
    type: str = None
) -> None:
    """
    For a given dataset and a certain minmax range, computes the budgets and precisions for
    num_points total experiments of lsh-mips. Returns a numpy array of budgets and their corresponding precision.

    :seed: A random seed
    :top_k: A list of how many candidates to return
    :generate_data: A name of data generating function
    :is_normalize: Whether to normalize atoms and signal
    :num_points: Number of points to plot on graph
    :hfunc_minmax: Minmax of number of hyperplanes
    :table_minmax: Minmax of table size.
    :type: Either symmetric or asymmetric
    """
    # generate the data
    if generate_data == ADVERSARIAL_CUSTOM:
        generate_data = generate_adversarial_data
    elif generate_data == NORMAL_CUSTOM:
        generate_data = generate_normal_data
    elif generate_data == NORMAL_PAPER:
        generate_data = generate_normal_paper
    else:
        raise NotImplementedError(f"{generate_data} is not implemented")
    #atoms, signal = generate_data(num_atoms=1000, len_signal=10000, seed=seed)
    atoms, signal = generate_data(num_atoms=100, len_signal=1000, seed=seed)

    # by default, the data is NOT normalized
    if is_normalize:
        atoms_norm = np.linalg.norm(atoms, axis=1, keepdims=True)
        signal_norm = np.linalg.norm(signal, keepdims=True)
        atoms /= atoms_norm
        signal /= signal_norm

    # setup experiments by defining the array of num_hfunc and num_tables.
    # This is analogous to setting up pairs of (a, b) in the paper.
    table_diff = table_minmax[1] - table_minmax[0] + 1
    num_hfuncs = np.linspace(hfunc_minmax[0], hfunc_minmax[1], num_points // table_diff).astype(np.int64)
    num_tables = np.linspace(table_minmax[0], table_minmax[1], table_diff).astype(np.int64)
    assert (len(num_hfuncs) * len(num_tables) == num_points)

    # running the experiments and concatenating the results in the budget, precisions list.
    # both lists should be of size num_points at the end of the experiment
    budgets = []
    precisions = []
    candidates = naive_mips(atoms, signal, top_k=top_k)
    for num_table in num_tables:
        for num_hfunc in num_hfuncs:
            print("     ... num_tables, num_hfunc: ", (num_table, num_hfunc))
            budget, _candidates = run_lsh_mips(
                type=type,
                data=atoms,
                num_hfunc=num_hfunc,  # this is analogous to "a"
                num_tables=num_table,  # this is analogous to "b"
                query=signal,
                top_k=top_k,
            )
            budgets.append(np.array(budget))
            precisions.append(
                np.array(compare_with_naive(candidates, _candidates, top_k))
            )
    return np.array(budgets), np.array(precisions)


