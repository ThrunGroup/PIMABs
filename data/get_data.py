from typing import Tuple
import numpy as np
import math
import os
import mnist

from utils.constants import (
    ADVERSARIAL_CUSTOM,
    NORMAL_CUSTOM,
    NORMAL_PAPER,
    UNIFORM_PAPER,
    YAHOO,
    NETFLIX,
    MOVIE_LENS,
    CLEAR_LEADER_HARD,
    CLEAR_LEADER_SOFT,
    NO_CLEAR_LEADER,
    MNIST_T,
)
from data.custom_data import generate_custom_data
from utils.utils import subset_2d


def get_data(
    num_atoms: int = 10 ** 3,
    len_signal: int = 10 ** 4,
    num_signals: int = 1,
    seed: int = 0,
    data_type: str = NORMAL_CUSTOM,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if data_type in [ADVERSARIAL_CUSTOM, NORMAL_CUSTOM, NORMAL_PAPER, UNIFORM_PAPER, CLEAR_LEADER_HARD, CLEAR_LEADER_SOFT, NO_CLEAR_LEADER]:
        return generate_custom_data(
            num_atoms=num_atoms,
            len_signal=len_signal,
            num_signals=num_signals,
            seed=seed,
            data_type=data_type,
        )
    elif data_type == NETFLIX:
        filename = os.path.join(
            os.path.join(os.path.dirname(__file__), "netflix"), "Movie_ratings.npy"
        )
        data = np.load(filename)
    elif data_type == MOVIE_LENS:
        filename = os.path.join(
            os.path.join(os.path.dirname(__file__), "movie_lens"),
            "movie_lens_1m_ratings.npy",
        )
        data = np.load(filename)
    elif data_type == MNIST_T:
        mndata = mnist.MNIST(os.path.join(os.path.dirname(__file__), "mnist"))
        images, labels = mndata.load_training()
        data = np.array(images)
        data = data.T
        num_atoms = 783
        num_signals = 1
    else:
        raise NotImplementedError(f"{data_type} is not implemented")

    assert (
        num_atoms + num_signals <= data.shape[0]
    ), "Number of atoms + number of signals given are greater than the number of rows of data"
    assert (
        len_signal <= data.shape[1]
    ), "Length of signal given is greater than the number of columns of data"

    signal_idcs = rng.choice(np.arange(data.shape[0]), num_signals, replace=False)
    sample_idcs = rng.choice(data.shape[1], len_signal)

    atom_idcs = rng.choice(np.setdiff1d(np.arange(data.shape[0]), signal_idcs), num_atoms, replace=False)
    atoms = subset_2d(data, atom_idcs, sample_idcs)
    signals = subset_2d(data, signal_idcs, sample_idcs)
    del data
    return atoms, signals