import numpy as np
from load_datasets import load_netflix, load_yahoo
from utils.constants import (
    NUM_NETFLIX_USERS,
    NUM_YAHOO_USERS,
    SAMPLE_SIZE_FOR_N,
)


def data_sanity_checks():
    # TODO: Also need to check the values!
    load_netflix()
    assert np.load("netflix_dataset_array.npy").shape == (
        SAMPLE_SIZE_FOR_N,
        NUM_NETFLIX_USERS,
    )
    load_yahoo()
    assert np.load("yahoo_dataset_array.npy").shape == (
        SAMPLE_SIZE_FOR_N,
        NUM_YAHOO_USERS,
    )


if __name__ == "__main__":

    data_sanity_checks()
    print("Checks passed")
