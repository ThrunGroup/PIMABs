import numpy as np

from mips_bandit import mips_bandit


class BanditMIPSAdd:
    """
    Supports adding new elements
    """

    def __init__(self):
        self.top_k_indices = None
        self.top_k_atoms = None

    def search_top_k(self,
                     atoms: np.ndarray,
                     signals: np.ndarray,
                     maxmin: np.ndarray,
                     bandit_alg: str,
                     epsilon: float,
                     delta: float,
                     num_best_atoms: int = 1,
                     seed: int = 0,
                     verbose: bool = False,
                     with_replacement: bool = False,
                     new_data: bool = False,
                     save_top_k: bool = True,
                     ):
        if new_data:
            assert self.top_k_atoms is not None, "There are no original best atoms saved."
            atoms = np.append(atoms, self.top_k_atoms)

        mips_results = mips_bandit(atoms=atoms,
                                   signals=signals,
                                   maxmin=maxmin,
                                   bandit_alg=bandit_alg,
                                   epsilon=epsilon,
                                   delta=delta,
                                   num_best_atoms=num_best_atoms,
                                   seed=seed,
                                   is_experiment=False,
                                   with_replacement=with_replacement,
                                   verbose=verbose,
                                   )

        if save_top_k:
            candidates_array, _ = mips_results[:2]  # mips_bandit returns up to five values
            self.top_k_atoms = atoms[candidates_array]

        return mips_results
