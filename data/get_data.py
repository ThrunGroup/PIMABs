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
    SIMPLE_SONG,
    SAMPLE_RATE,
    SECONDS_PER_MINUTE,
)
from data.custom_data import generate_custom_data
from data.simple_song import create_song, create_note_waves, create_sin_wave
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
    elif data_type == SIMPLE_SONG:
        assert num_atoms >= 6, "Need at least 6 atoms for the simple song's notes"
        num_mins = 10  # Creates 26,460,000-dimensional data
        song = create_song(num_minutes=num_mins)
        signals = song.reshape(1, -1)
        num_signals = 1

        C0, E0, G0, C1, E1, G1 = create_note_waves()
        G1_freq = 392
        atoms = np.vstack([C0, E0, G0, C1, E1, G1])
        for i in range(7, num_atoms):
            note_freq = math.floor(G1_freq * 2 ** ((i - 6) / 12))
            print("Creating note with frequency", note_freq)
            extraneous_note = create_sin_wave(frequency_multiplier=note_freq, sampling_frequency=SAMPLE_RATE, num_samples=SAMPLE_RATE * SECONDS_PER_MINUTE*num_mins)
            atoms = np.vstack([atoms, extraneous_note])

        # sample_idcs = rng.choice(signals.shape[1], len_signal)
        # TODO(@motiwari): This is not the length of sample_idcs!
        import ipdb; ipdb.set_trace()
        return atoms, signals
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