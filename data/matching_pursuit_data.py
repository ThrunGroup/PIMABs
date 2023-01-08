import numpy as np
import os
from utils.utils import subset_2d


def get_environmental_sound_data(
        num_atoms: int = 10 ** 3,
        len_signal: int = 10 ** 4,
        num_signals: int = 1,
        seed: int = 0,
):
    rng = np.random.default_rng(seed)

    filename = os.path.join(
        os.path.join(os.path.dirname(__file__), "environmental_sound"),
        "environmental_sound.npy",
    )
    data = np.load(filename)

    # Set the size of the arrays and the frequency range
    fmin = 20  # Minimum frequency in Hz
    fmax = 8000  # Maximum frequency in Hz

    assert (
            num_signals <= data.shape[0]
    ), f"Number of signals given are greater than the number of rows of signals ({len(data)})"
    assert (
            len_signal < data.shape[1]
    ), f"Length of signal given is greater than the number of columns of signals ({data.shape[1]})"
    assert (
            num_atoms < fmax // 2
    ), f"Number of atoms given is greater than the maximum number of waves ({fmax // 2})"

    # Generate sine waves with frequencies ranging from fmin to fmax
    frequencies = np.linspace(fmin, fmax, num_atoms // 2)  # Array of frequencies in Hz
    t = np.arange(len_signal)  # Array of time points
    sin_waves = np.sin(2 * np.pi * frequencies[:, np.newaxis] * t / len_signal)  # Generate sine waves
    cos_waves = np.cos(2 * np.pi * frequencies[:, np.newaxis] * t / len_signal)  # Generate sine waves
    basis_functions = np.concatenate([sin_waves, cos_waves])

    signal_idcs = rng.choice(np.arange(0, data.shape[0]), num_signals)
    sample_idcs = rng.choice(data.shape[1], len_signal)

    # atoms = subset_2d(basis_functions, np.arange(num_atoms), sample_idcs)
    atoms = basis_functions[:, :len_signal]
    # signals = subset_2d(data, signal_idcs, sample_idcs)
    signals = data[:, :len_signal]

    # print(signals[0][:100])

    del data
    return atoms, signals
