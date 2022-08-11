from load_yahoo_music_data import load_yahoo_music_data
from utils.constants import (
    NUM_YAHOO_USERS,
    NUM_YAHOO_ITEMS,
)


def run_sanity_checks():
    assert load_yahoo_music_data().shape == (NUM_YAHOO_USERS, NUM_YAHOO_ITEMS)


if __name__ == "__main__":
    run_sanity_checks()
