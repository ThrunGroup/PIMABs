from typing import Tuple
import numpy as np
import math
import os

from utils.utils import subset_2d
from data.custom_data import generate_custom_data, create_toy_waves
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
    TOY_SINE,
    TOY_DISCRETE,
)


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
    elif data_type is NETFLIX:
        filename = os.path.join(
            os.path.join(os.path.dirname(__file__), "netflix"), "Movie_ratings.npy"
        )
        data = np.load(filename)
    elif data_type is MOVIE_LENS:
        filename = os.path.join(
            os.path.join(os.path.dirname(__file__), "movie_lens"),
            "movie_lens_1m_ratings.npy",
        )
        data = np.load(filename)
    elif data_type is TOY_SINE:
        atoms, signal = create_toy_waves(num_atoms, len_signal)
        return atoms, np.array([signal])
    else:
        raise NotImplementedError(f"{data_type} is not implemented")

    assert (
        num_atoms + num_signals < data.shape[0]
    ), "Number of atoms + number of signals given are greater than the number of rows of data"
    assert (
        len_signal < data.shape[1]
    ), "Length of signal given is greater than the number of columns of data"

    signal_idcs = rng.choice(np.arange(num_atoms, data.shape[0] - 1), num_signals)
    sample_idcs = rng.choice(data.shape[1], len_signal)

    atoms = subset_2d(data, np.arange(num_atoms), sample_idcs)
    signals = subset_2d(data, signal_idcs, sample_idcs)
    del data
    return atoms, signals