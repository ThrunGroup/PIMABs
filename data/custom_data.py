import numpy as np
import math
from typing import Tuple

from utils.constants import (
    ADVERSARIAL_CUSTOM,
    NORMAL_CUSTOM,
    NORMAL_PAPER,
    UNIFORM_PAPER,
    COR_NORMAL_CUSTOM,
)


def generate_custom_data(
    num_atoms: int = 10**3,
    len_signal: int = 10**4,
    num_signals: int = 1,
    seed: int = 0,
    data_type: str = NORMAL_CUSTOM,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a custom dataset described in a bandit approach to MIPS paper.

    :param num_atoms: Number of atoms(arms)
    :param len_signal: A size of signal vector
    :param num_signals: Number of signals
    :param seed: A random seed
    :param data_type: The name of distribution from which data are drawn
    :return: A synthetic dataset
    """
    rng = np.random.default_rng(seed)
    atoms_array = np.empty((num_atoms, len_signal))
    if (
        data_type is UNIFORM_PAPER
    ):  # Hardcode so that variance proxy(sub-gaussian distribution) is 1
        signal = rng.uniform(low=0, high=2, size=(num_signals, len_signal))
        atoms_array = rng.uniform(low=0, high=2, size=(num_atoms, len_signal))
    elif (
        data_type is NORMAL_PAPER
    ):  # Hardcode so that variance proxy(sub-gaussian distribution) is 1
        signal = rng.normal(loc=2, size=(num_signals, len_signal))
        atoms_array = rng.normal(size=(num_atoms, len_signal))
    elif data_type is ADVERSARIAL_CUSTOM:
        atoms_array = np.zeros((num_atoms, len_signal))
        p = rng.random(num_atoms)
        for idx in range(num_atoms):
            atoms_array[idx][: math.floor(p[idx] * len_signal)] = 1
        signal = np.ones((num_signals, len_signal))
    elif (
        data_type is NORMAL_CUSTOM
    ):  # Hardcode so that variance proxy(sub-gaussian distribution) is 1
        means = rng.normal(size=num_atoms)
        atoms_array = np.empty((num_atoms, len_signal))
        for idx in range(num_atoms):
            atoms_array[idx] = rng.normal(loc=means[idx], size=len_signal)
        signal = rng.normal(loc=rng.normal(), size=(num_signals, len_signal))
    elif (
            data_type is COR_NORMAL_CUSTOM
    ):  # Hardcode so that variance proxy(sub-gaussian distribution) is 1
        correlations = rng.normal(size=(num_signals + 1, num_atoms))
        correlations = np.abs(correlations) / np.sum(np.abs(correlations), axis=0, keepdims=True)
        atoms_array = np.zeros((num_atoms, len_signal))
        signal = rng.normal(loc=rng.normal(), size=(num_signals + 1, len_signal))  # To include independet normal rv
        for idx in range(num_atoms):
            for cor_idx in range(num_signals):
                atoms_array[idx] += correlations[cor_idx, idx] * signal[cor_idx]
        atoms_array += 3 * np.expand_dims(rng.normal(size=num_atoms), axis=1)
        signal = signal[:-1, :]
    else:
        raise NotImplementedError(f"{data_type} is not implemented.")
    return atoms_array, signal
