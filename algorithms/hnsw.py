import numpy as np
import hnswlib

from data.get_data import get_data
from utils.constants import (
    NORMAL_CUSTOM,
    NORMALIZED_MAXMIN,
    ACTION_ELIMINATION,
)
from mips_bandit import mips_bandit


def compare(seed, n, d):
    atoms, signals = get_data(
        num_atoms=n,
        len_signal=d,
        data_type=NORMAL_CUSTOM,
        num_signals=3,
        seed=seed
    )

    index = hnswlib.Index(space="ip", dim=d, useNormFactor_=True)

    # Initializing index - the maximum number of elements should be known beforehand
    index.init_index(max_elements=n, ef_construction=200, M=16)

    # Element insertion (can be called several times):
    ids = np.arange(n)
    index.add_items(atoms, ids)

    labels, distances, budget1 = index.knn_query(signals, k=4)
    labels.sort()

    index2 = hnswlib.Index(space="ip", dim=d, useNormFactor_=False)

    # Initializing index - the maximum number of elements should be known beforehand
    index2.init_index(max_elements=n, ef_construction=200, M=16)

    # Element insertion (can be called several times):
    ids2 = np.arange(n)
    index2.add_items(atoms, ids2)

    labels2, distances, budget2 = index2.knn_query(signals, k=4)
    labels2.sort()

    candidates_array, budget_array = mips_bandit(
        atoms=atoms,
        signals=signals,
        maxmin=NORMALIZED_MAXMIN,
        bandit_alg=ACTION_ELIMINATION,
        epsilon=0.01,
        delta=0.001,
        num_best_atoms=4,
        is_experiment=False,
        with_replacement=False,
    )
    candidates_array.sort()

    print(f"seed: {seed}")
    print("NAHNSW accuracy: ", np.mean(labels == candidates_array))
    print("HNSW accuracy  : ", np.mean(labels2 == candidates_array))
    print("===============")


n = 1000
d = 100

for seed in range(10):
    compare(seed, n, d)
