# Matching Pursuit and Permutation-Invariant Multi-Armed Bandits (PIMABs)

## Usage

Run `python mp_tests.py --seed 0 -d 1000` to run the tests in `mp_tests.py`. This should provide some sanity checks
that Bandit Matching Pursuit is returning the correct results.

## Description of files

- `matching_pursuit_algs.py`: Contains the two algorithms for the matching pursuit problem, canonical Matching Pursuit and Bandit Matching Pursuit
- `mp_tests.py`: 
- `utils.py`: some helper functions used to run experiments
- `constants.py`: for some constants used in the various algorithms
- `Book1.xlsx`: I'm not really sure, but it looks like this contained some experimental results for running BanditMP for
various dataset sizes over 10 runs. The scaling does not look linear (which is a good thing), though I can't remember
exactly how to recreate these results.
- `data/`: Contains the MNIST dataset
- `exps/`: Contains helper code for running many experiments in serial or in parallel, and then saving and analyzing the results. This subdirectory's code hasn't been cleaned yet