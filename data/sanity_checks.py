from load_datasets import load_netflix, load_yahoo
from utils.constants import (
    NUM_NETFLIX_USERS,
    NUM_NETFLIX_MOVIES,
    NUM_YAHOO_USERS,
    NUM_YAHOO_ITEMS,
)


def data_sanity_checks():
    # TODO: Also need to check the values!
    assert load_netflix().shape == (NUM_NETFLIX_USERS, NUM_NETFLIX_MOVIES)
    assert load_yahoo().shape == (NUM_YAHOO_USERS, NUM_YAHOO_ITEMS)


if __name__ == "__main__":


    data_sanity_checks()
    print("Checks passed")
