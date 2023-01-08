import numpy as np
import os
from glob import glob
import scipy
import tqdm


def main(num_signals: int = 20, num_dimensions: int = 10 ** 5):
    current_directory = os.path.dirname(__file__)
    audio_directory = os.path.join(current_directory, "archive", "audio", "audio", "44100")
    dataset = np.empty((num_signals, num_dimensions))
    num_points_per_file = 220500
    num_files = num_dimensions // num_points_per_file + 1

    print("Making a dataset...")
    for signal_index in tqdm.tqdm(range(num_signals), total=num_signals):
        files = glob(os.path.join(audio_directory, f"*-{signal_index}.wav"))[:num_files]

        signal_data = np.empty(num_files * num_points_per_file)

        for (i, file) in enumerate(files):
            _, audio_data = scipy.io.wavfile.read(file)
            start_index = i * num_points_per_file
            end_index = start_index + num_points_per_file
            signal_data[start_index:end_index] = np.array(audio_data)

        dataset[signal_index] = signal_data[:num_dimensions]

    np.save(os.path.join(current_directory, "environmental_sound.npy"), dataset)


def dictionary_learning(num_data: int = 50, num_dimensions: int = 10 ** 5):
    current_directory = os.path.dirname(__file__)
    audio_directory = os.path.join(current_directory, "archive", "audio", "audio", "44100")
    dataset = np.empty((num_data, num_dimensions))
    num_points_per_file = 220500
    num_files = num_dimensions // num_points_per_file + 1

    print("Making a dataset...")
    for signal_index in tqdm.tqdm(range(num_data), total=num_data):
        files = glob(os.path.join(audio_directory, f"*-{signal_index}.wav"))[:num_files]

        signal_data = np.empty(num_files * num_points_per_file)

        for (i, file) in enumerate(files):
            _, audio_data = scipy.io.wavfile.read(file)
            start_index = i * num_points_per_file
            end_index = start_index + num_points_per_file
            signal_data[start_index:end_index] = np.array(audio_data)

        dataset[signal_index] = signal_data[:num_dimensions]

    np.save(os.path.join(current_directory, "environmental_sound_dictionary_learning.npy"), dataset)


if __name__ == "__main__":
    # main()
    dictionary_learning()
