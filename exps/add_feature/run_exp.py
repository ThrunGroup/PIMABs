"""
1. Get Netflix dataset
2. Split it into halves
3. Run A + B first
4. Run A and then B
5. Compare the top k
"""
from data.get_data import get_data
from utils.constants import NETFLIX


def run_exp():
    # Get Netflix dataset

    atoms, signals = get_data(
        num_atoms=10000,
        len_signal=50000,
        num_signals=10,
        seed=0,
        data_type=NETFLIX,
    )


if __name__ == "__main__":
    run_exp()