import numpy as np
from matching_pursuit import matching_pursuit, b_matching_pursuit
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Solve MIPS problem")
    parser.add_argument("--get_ground_truth", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--seed", default=0)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--p", type=float)
    return parser.parse_args()

def test_case_1(args):
    dim = 5
    signal = np.ones(dim)
    atoms = np.array([
        np.ones(dim) / np.sqrt(5),
        [0, 1, 0, 0, 1] / np.sqrt(2),
    ])
    coefs, counter = matching_pursuit(signal, atoms)
    assert np.allclose(coefs, np.array([2.23606798, 0]))
    assert counter == 10

    b_coefs, b_counter = b_matching_pursuit(signal, atoms, args)
    assert np.allclose(b_coefs, np.array([2.23606798, 0]))
    assert b_counter == 10


def test_case_2():
    signal = np.array([1., 0, 1, 0, 1, 0, 1])
    atoms = np.array([
        np.array([1., 0, 1, 0, 1, 0, 1]) / 2,
        np.array([0, 1., 0, 1, 0, 1, 0]) / np.sqrt(3),
    ])
    coefs, counter = matching_pursuit(signal, atoms)
    assert np.allclose(coefs, np.array([2, 0]))
    assert counter == 14

    b_coefs, b_counter = b_matching_pursuit(signal, atoms, args)
    assert np.allclose(b_coefs, np.array([2, 0]))
    assert b_counter == 14

def test_case_3():
    signal = np.array([1., 1, 1, 0, 1, 0, 1])
    atoms = np.array([
        np.array([1., 0, 1, 0, 1, 0, 1]) / 2,
        np.array([0, 1., 0, 1, 0, 1, 0]) / np.sqrt(3),
        np.array([0, 1., 0, 0, 0, 0, 0]),
    ])
    coefs, counter = matching_pursuit(signal, atoms)
    assert np.allclose(coefs, np.array([2, 0, 1]))
    assert counter == 42

    b_coefs, b_counter = b_matching_pursuit(signal, atoms, args)
    assert np.allclose(b_coefs, np.array([2, 0, 1]))
    assert b_counter == 42


def main(args):
    np.random.seed(args.seed)
    test_case_1(args)
    test_case_2()
    test_case_3()


if __name__ == '__main__':
    args = get_args()
    main(args)
    print("All tests pass")