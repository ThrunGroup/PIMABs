import numpy as np

from algorithms.mips_bandit import mips_bandit


class BanditMIPSAdd:
    """
    Supports adding new elements
    """

    def __init__(self):
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
                     is_addition: bool = False,
                     save_top_k: bool = True,
                     ):
        """
        Run BanditMIPS with an extended feature that lets you solve MIPS with a new data stream

        :param atoms: 2d array that contains atoms in rows. Atoms are the vectors one of which will be selected
                        as the vector with maximum inner product with a signal vector.
        :param signals: A matrix whose each row contains a signal vector.
        :param maxmin: A tuple with two elements which are the maximum and minimum of data_atoms array
        :param bandit_alg: A name of multi-armed bandit algorithm
        :param epsilon: A maximum additive error between the true best arm and the best arm identified by our
                        bandit algorithm
        :param delta: A probability that the difference between the reward of the true best arm and the identified
                      best arm is beyond epsilon
        :param num_best_atoms: Number of atoms that bandit algorithm identify as top/best arms.
        :param seed: A random seed
        :param verbose: Verbosity of function
        :param with_replacement: Whether to sample with replacement
        :param is_addition: Whether the dataset is new
        :param save_top_k: Whether to save the current best atoms. Set it false if you're sure there won't be new
                            data stream
        :return: The best atoms that have maximum inner product with a signal vector
        """

        if is_addition:
            assert self.top_k_atoms is not None, "There are no original best atoms saved."
            atoms = np.concatenate([atoms, self.top_k_atoms])

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

        candidates_array, _ = mips_results[:2]  # mips_bandit returns up to five values
        top_k_atoms = atoms[candidates_array].reshape(-1, signals.shape[-1])

        if save_top_k:
            self.top_k_atoms = top_k_atoms

        return top_k_atoms
