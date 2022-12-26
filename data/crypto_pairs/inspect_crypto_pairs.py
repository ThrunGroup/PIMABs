import pandas as pd
import numpy as np
import os
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


def main(with_volume: bool):
    """
    Visualise the distribution crypto pairs dataset
    """
    dataset_file = "crypto_pairs_1m_dimensions.npy"
    dataset = np.load(dataset_file)

    # Draw a violin graph for every atom
    fig, ax = plt.subplots()
    for i in range(0, len(dataset)):
        ax.violinplot(dataset=dataset[i], positions=[i])

    plt.show()

    # Draw a violin graph for 10 atoms (clipped under 10)
    fig, ax = plt.subplots()
    for i in range(0, 10):
        ax.violinplot(dataset=np.clip(dataset[i], a_min=None, a_max=10), positions=[i], showmeans=True,
                      showextrema=True)

    plt.show()



if __name__ == "__main__":
    main(with_volume=True)
