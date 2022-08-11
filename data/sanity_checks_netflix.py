from load_netflix_data import load_netflix_data
from utils.constants import (
    NUM_NETFLIX_USERS,
    NUM_NETFLIX_MOVIES,
)


def run_sanity_checks():
    assert load_netflix_data().shape == (NUM_NETFLIX_USERS, NUM_NETFLIX_MOVIES)


if __name__ == "__main__":
    run_sanity_checks()
