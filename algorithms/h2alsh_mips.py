import numpy as np
from math import sqrt, log
from qalsh import QALSH
from typing import Tuple

class H2ALSH_MIPS:
    """
    Python Implementation of Homocentric Hypersphere Asymmetric Locally Sensitive Hashing scheme for MIPS, see
    https://dl.acm.org/doi/pdf/10.1145/3219819.3219971 for the details.
    """
    def __init__(self, atoms: np.ndarray, delta: float = 0.1, c0: float = 2.0, c: float = 0.9,  N0: int = 500):
        """

        :param atoms: Atoms/Objects that we consider in MIPS problem
        :param delta: Error probability (QALSH solves c-ANN problem with 1/2 - delta probability)
        :param c0: denotes c0 in c0-ANN problem. See 2.1 in the paper for the definition of c0-ANN.
        :param c: denotes c in c-AMIP problem. See 2.1 in the paper for the definition of c-AMIP.
        :param N0: Threshold of the size of partition where the MIP atom is searched through linear scanning
        """
        assert N0 > 100, "N0 should be greater than 100 since QALSH can solve NN for more than 100 atoms"
        # Store parameters as self variables
        self.atoms = atoms
        self.delta = delta
        self.c0 = c0
        self.c = c
        self.N0 = N0

        # Partitioning atoms
        b = sqrt(1 - (1-c)/(c0 ** 2 - c))  # Partition ratio, see 3.3 in the paper for the formula
        atoms_norm = np.linalg.norm(atoms, axis=1)
        sort_idcs = np.argsort(atoms_norm)[::-1]  # Sort in the ascending order
        j = 0
        i = 0
        partitions = []  # List of partitions
        partitions_norms = []  # List of maximum norm of partitions
        partitions_idcs = []
        linear_scan_mask = []
        self.qalsh_dict = {}  # Build hash tables for partition whose size is greater than N0
        while j <= atoms.shape[0] - 1:  # atoms.shape[0] is the # of atoms
            ub = atoms_norm[sort_idcs[j]]
            current_partition = []
            current_partition_idcs = []
            while (j <= atoms.shape[0] - 1) and (atoms_norm[sort_idcs[j]] > b * ub):
                transformed_atom = np.concatenate((atoms[sort_idcs[j]], [sqrt(ub ** 2 - atoms_norm[sort_idcs[j]] ** 2)]))
                current_partition.append(transformed_atom)
                current_partition_idcs.append(sort_idcs[j])
                j += 1

            # If the size of partition is small enough(<= N0), then naively compute the inner products of atoms in
            # the partition to find MIP atom
            if len(current_partition) <= N0:
                linear_scan_mask.append(1)
            else:
                linear_scan_mask.append(0)
                qalsh = QALSH(np.array(current_partition), delta=delta, approx_const=c0)
                qalsh.generate_hash_tables()
                self.qalsh_dict[i] = qalsh
            i += 1

            partitions.append(np.array(current_partition))
            partitions_idcs.append(np.array(current_partition_idcs))
            partitions_norms.append(ub)

        self.linear_scan_mask = np.array(linear_scan_mask)
        self.S = partitions
        self.M = partitions_norms
        self.partitions_idcs = partitions_idcs

    def mip_search(self, query, top_k: int = 1):
        query_norm = np.linalg.norm(query)
        candidates = []
        candidates_ip = []  # Inner product between candidates and the query
        mip_value = -float('inf')
        sample_complexity = 0
        for i in range(len(self.S)):
            ub = self.M[i] * query_norm
            if ub <= mip_value:
                break
            if self.linear_scan_mask[i] == 1:
                current_ip = self.atoms[self.partitions_idcs[i]] @ query
                current_candidates = np.argsort(current_ip)[::-1][:top_k]
                current_candidates_ip = current_ip[current_candidates]
                sample_complexity += len(self.partitions_idcs[i]) * len(query)
            else:
                alpha = self.M[i] / query_norm
                transformed_query = np.concatenate((alpha * query, [0]))
                current_candidates_idcs, current_sample_complexity = self.qalsh_dict[i].nn_search(transformed_query, top_k)
                current_candidates = self.partitions_idcs[i][current_candidates_idcs]
                current_candidates_ip = self.atoms[current_candidates] @ query
                sample_complexity += current_sample_complexity + len(current_candidates) * len(query)

            new_candidates_idcs = np.argsort(np.concatenate((candidates_ip, current_candidates_ip)))[::-1][:top_k]
            candidates = np.concatenate((candidates, current_candidates))[new_candidates_idcs]
            candidates_ip = np.concatenate((candidates_ip, current_candidates_ip))[new_candidates_idcs]
            mip_value = candidates_ip[-1]
        return candidates, sample_complexity

    def mip_search_qeuries(self, queries: np.ndarray, top_k: int = 1) -> Tuple[np.ndarray, int]:
        sample_complexity = 0
        candidates_array = []
        for query in queries:
            curr_candidates, curr_complexity = self.mip_search(query, top_k)
            candidates_array.append(curr_candidates)
            sample_complexity += curr_complexity
        return np.array(candidates_array), sample_complexity


def toy_experiment_h2alsh(seed: int = 0):
    np.random.seed(seed)
    atoms = np.random.normal(size=(300000, 10))
    query = np.random.normal(size=10)
    top_k = 1
    naive_ans = np.argsort(atoms@query)[::-1][:top_k]

    h2alsh = H2ALSH_MIPS(atoms=atoms, c0=1.5, delta=0.02)
    mips = h2alsh.mip_search(query, top_k=5)
    sample_complexity_ratio = mips[1] / (atoms.shape[0] * atoms.shape[1])

    print(
        f"It is {naive_ans == mips[0]} that naive MIP search accords with H2ALSH MIP search. "
        f"The sample complexity of H2ALSH is {sample_complexity_ratio * 100}% of one of naive MIP search"
    )


if __name__ == "__main__":
    toy_experiment_h2alsh()













