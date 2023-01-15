import numpy as np
import soundfile as sf
from utils.constants import (
    SAMPLE_RATE,
    SECONDS_PER_MINUTE,
)


def create_sin_wave(frequency_multiplier: int, sampling_frequency: int = SAMPLE_RATE, num_samples: int = SAMPLE_RATE * SECONDS_PER_MINUTE):
    """
    Create a discretized sine wave with the ith value being sin(frequency_multiplier * i / sampling_frequency)

    Sampling frequency should be given in Hz
    :param frequency_multiplier: Frequency multiplier
    :param sampling_frequency: Sampling frequency
    :param num_samples: Number of samples
    :return: Array of sine wave values
    """
    assert frequency_multiplier > 0, "Not implemented for <= 0 frequencies yet"
    phase = frequency_multiplier * np.arange(num_samples) / sampling_frequency
    return np.sin(phase)


def create_cos_wave(frequency_multiplier: int, sampling_frequency: int = SAMPLE_RATE, num_samples: int = SAMPLE_RATE * SECONDS_PER_MINUTE):
    """
    Create a discretized cosine wave with the ith value being cos(frequency_multiplier * i / sampling_frequency)

    Sampling frequency should be given in Hz

    :param frequency_multiplier: Frequency multiplier
    :param sampling_frequency: Sampling frequency
    :param num_samples: Number of samples
    :return: Array of cosine wave values
    """
    assert frequency_multiplier > 0, "Not implemented for <= 0 frequencies yet"
    phase = frequency_multiplier * np.arange(num_samples) / sampling_frequency
    return np.cos(phase)


def stack(arr_, num):
    """
    Convenience function to horizontally stack an array (increase its width) by num times

    :param arr_: Array to stack horizontally
    :param num: Number of times to duplicate the array
    :return: Horizontally replicated array
    """
    return np.hstack([arr_ for _ in range(num)])


def create_song(num_minutes: int = 10):
    """
    Creates a simple song of length num_minutes minutes.
    The song is created by playing C0 + 2*E0 + 3*G0 for 1 minute, then 3*G0 + 2.5*C1 + 1.5*E1 for 1 minute, and
    repeatedly alternating.

    :param num_minutes: Number of minutes the song should play
    :return:
    """

    # Note: Need to use 2*\pi exactly here! Using 6.28 means that the waves won't go through
    # full cycles each second, which means stacking them 10 times will give different answers from just continuing them
    two_pi = 2 * np.pi
    C0 = create_sin_wave(frequency_multiplier=256 * two_pi, sampling_frequency=SAMPLE_RATE,
                         num_samples=SAMPLE_RATE * SECONDS_PER_MINUTE)
    E0 = create_sin_wave(frequency_multiplier=330 * two_pi, sampling_frequency=SAMPLE_RATE,
                         num_samples=SAMPLE_RATE * SECONDS_PER_MINUTE)
    G0 = create_sin_wave(frequency_multiplier=392 * two_pi, sampling_frequency=SAMPLE_RATE,
                         num_samples=SAMPLE_RATE * SECONDS_PER_MINUTE)

    C1 = create_sin_wave(frequency_multiplier=2 * 256 * two_pi, sampling_frequency=SAMPLE_RATE,
                         num_samples=SAMPLE_RATE * SECONDS_PER_MINUTE)
    E1 = create_sin_wave(frequency_multiplier=2 * 330 * two_pi, sampling_frequency=SAMPLE_RATE,
                         num_samples=SAMPLE_RATE * SECONDS_PER_MINUTE)
    G1 = create_sin_wave(frequency_multiplier=2 * 392 * two_pi, sampling_frequency=SAMPLE_RATE,
                         num_samples=SAMPLE_RATE * SECONDS_PER_MINUTE)

    song = np.array([])
    assert int(num_minutes) == num_minutes, "Must pass integer number of num_minutes"
    for i in range(int(num_minutes)):
        if i % 2 == 0:
            song = np.hstack([song, C0 + 2*E0 + 3*G0])
        else:
            song = np.hstack([song, 3*G0 + 2.5*C1 + 1.5*E1])

    return song

def main():
    song = create_song(num_minutes=10)
    from algorithms.action_elimination import action_elimination

    signals = song.reshape(1, -1)

    frequencies = [256 * 6.28, 330 * 6.28, 392 * 6.28, 2 * 256 * 6.28, 2 * 330 * 6.28, 2 * 392 * 6.28]
    atoms = np.array([create_sin_wave(frequency_multiplier=f, sampling_frequency=SAMPLE_RATE,
                                      num_samples=SAMPLE_RATE * SECONDS_PER_MINUTE * 10) for f in frequencies])
    atom, budget = action_elimination(stack(atoms, 2), stack(signals, 2), var_proxy=0.1, epsilon=0, delta=0.0003, abs=True,
                       batch_size=300)
    print(atom, budget)
    assert(atom[0, 0] == 2)

if __name__ == "__main__":
    main()