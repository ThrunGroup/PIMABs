import numpy as np
import math
from typing import Tuple

def generate_adversarial_data(
    num_atoms: int = 10 ** 3, len_signal: int = 10 ** 4, seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an adversarial dataset with given parameters. Refer to experiment (1) in "shorturl.at/adPR3".

    :param num_atoms: Number of atoms (arms)
    :param len_signal: A size of signal vector
    :param seed: A random seed
    :return: An adversarial dataset for MIPS problem. Returns atoms
    (2d-array) and a signal vector(1d-array).
    """
    rng = np.random.default_rng(seed)
    atoms_array = np.zeros((num_atoms, len_signal))
    p = rng.random(num_atoms)
    for idx in range(num_atoms):
        atoms_array[idx][: math.floor(p[idx] * len_signal)] = 1
    signal = np.ones(len_signal)
    return atoms_array, signal


def generate_normal_data(
    num_atoms: int = 10 ** 3, len_signal: int = 10 ** 4, seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a normal distributed rewards.

    :param num_atoms: Number of atoms(arms)
    :param len_signal: A size of signal vector
    :param seed: A random seed
    :return: An adversarial dataset for MIPS problem. Returns atoms(2d-array) and a signal vector(1d-array).
    """
    rng = np.random.default_rng(seed)
    means = rng.normal(size=num_atoms)
    atoms_array = np.empty((num_atoms, len_signal))
    for idx in range(num_atoms):
        atoms_array[idx] = rng.normal(loc=means[idx], size=len_signal)
    # signal = rng.normal(loc=0, size=len_signal)
    signal = np.ones(len_signal)
    return atoms_array, signal

def generate_normal_paper(
    num_atoms: int = 10 ** 3, len_signal: int = 10 ** 4, seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return np.random.normal(size=(num_atoms, len_signal)), np.ones(len_signal)
