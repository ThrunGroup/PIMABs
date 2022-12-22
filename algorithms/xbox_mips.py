import numpy as np
from numpy import ndarray

from qalsh import QALSH
from typing import Tuple


class XBOX_MIPS:
    def __init__(self, atoms: np.ndarray, delta: float = 0.1, approx_const: float = 2.0):
        """
        XBOX transformation to reduce MIPS to NNS problem. And then apply QALSH algorithm to find Nearest Neighbors.

        See the comments of init function of the QALSH class in qalsh.py if you're looking for the description of
        parameters.
        """
        atoms_norm = np.linalg.norm(atoms, axis=1)
        M = np.max(atoms_norm)
        self.transformed_atoms = np.concatenate((atoms, np.sqrt(M ** 2 - atoms_norm ** 2)), axis=1)
        self.delta = delta
        self.approx_const = approx_const
        self.qalsh = QALSH(atoms=atoms, delta=delta, approx_const=approx_const)

    def nn_search(
            self, query: np.ndarray, top_k: int = 1
    ) -> Tuple[np.ndarray, int]:
        transformed_query = np.concatenate((query, [0]))
        return self.qalsh.nn_search(transformed_query, top_k)

    def nn_search_qeuries(self, queries: np.ndarray, top_k: int = 1) -> Tuple[np.ndarray, int]:
        sample_complexity = 0
        candidates_array = []
        for query in queries:
            curr_candidates, curr_complexity = self.nn_search(query, top_k)
            candidates_array.append(curr_candidates)
            sample_complexity += curr_complexity
        return np.array(candidates_array), sample_complexity




