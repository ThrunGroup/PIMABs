import pandas as pd
import numpy as np
import os
import tqdm


def main(with_volume: bool = True):
    """
    Make a dataset for 105 crypto pairs with 1M dimensions
    """
    directory_path = os.path.dirname(os.path.abspath(__file__))
    dataset_directory = os.path.join(directory_path, "archive")
    num_atoms = 105
    if with_volume:
        min_file_size = 16 * 1024 * 1024  # 16MB
    else:
        min_file_size = 19 * 1024 * 1024  # 22MB
    columns = ["open", "close", "high", "low"]
    if with_volume:
        columns += "volume",  # volume columns has unexceptionally large values
    num_max_dimensions = 10 ** 6
    num_max_timestamps = num_max_dimensions // len(columns)

    files = []

    for file in os.listdir(dataset_directory):
        file_path = os.path.join(dataset_directory, file)

        if os.path.getsize(file_path) > min_file_size:
            files.append(file_path)

    files.sort(key=lambda file: os.path.getsize(file))
    files = files[:num_atoms]  # Only keep the crypto pairs that have more than 1M dimensions
    dataset = np.empty((len(files), num_max_dimensions))

    print("Making a crypto pairs dataset...")
    for (i, file) in tqdm.tqdm(enumerate(files), total=len(files)):
        df = pd.read_csv(file)
        dataset[i] = df[columns][:num_max_timestamps].values.flatten()

    np.save(os.path.join(os.getcwd(), "crypto_pairs_1m_dimensions.npy"), dataset)


if __name__ == "__main__":
    main(with_volume=False)
