import argparse
import numpy as np
from matching_pursuit_algs import matching_pursuit, bandit_matching_pursuit
from utils.utils import get_args


def test_case_1(args: argparse.Namespace) -> None:
    """
    Test case. Create a 5-dimensional signal vector that is all ones, and two atoms: one that is parallel to the signal
    and one that is not.

    Verify -- for both Matching Pursuit and Bandit Matching Pursuit -- that the only non-zero coefficient is for the first atom, which can completely express
    the signal.

    Also assert that each algorithm takes only 10 coordinate-wise multiplications to get the correct answer.

    :param args: Experiment arguments. Only max_iter, tol, residue, p, and batch_size are used
    :return: None
    """
    dim = 5
    signal = np.ones(dim)
    atoms = np.array(
        [
            np.ones(dim) / np.sqrt(5),
            [0, 1, 0, 0, 1] / np.sqrt(2),
        ]
    )
    coefs, counter = matching_pursuit(signal, atoms, args)
    assert np.allclose(coefs, np.array([2.23606798, 0]))
    assert counter == 10

    b_coefs, b_counter = bandit_matching_pursuit(signal, atoms, args)
    assert np.allclose(b_coefs, np.array([2.23606798, 0]))
    assert b_counter == 10


def test_case_2(args: argparse.Namespace) -> None:
    """
    Create a slightly more complicated 7-dimensional signal vector, and two atoms, one of which is parallel to the signal and the
    other of which is orthogonal to the signal.

    Verify -- for both Matching Pursuit and Bandit Matching Pursuit -- that the only non-zero coefficient is for the first atom, which can completely express
    the signal.

    Also assert that each algorithm takes only 14 coordinate-wise multiplications to get the correct answer.

    :param args: Experiment arguments. Only max_iter, tol, residue, p, and batch_size are used
    :return: None
    """
    signal = np.array([1.0, 0, 1, 0, 1, 0, 1])
    atoms = np.array(
        [
            np.array([1.0, 0, 1, 0, 1, 0, 1]) / 2,
            np.array([0, 1.0, 0, 1, 0, 1, 0]) / np.sqrt(3),
        ]
    )
    coefs, counter = matching_pursuit(signal, atoms, args)
    assert np.allclose(coefs, np.array([2, 0]))
    assert counter == 14

    b_coefs, b_counter = bandit_matching_pursuit(signal, atoms, args)
    assert np.allclose(b_coefs, np.array([2, 0]))
    assert b_counter == 14


def test_case_3(args: argparse.Namespace) -> None:
    """
    Create a slightly more complicated 7-dimensional signal vector, and three atoms, two of which can express the signal.

    Verify -- for both Matching Pursuit and Bandit Matching Pursuit -- correct answer is returned in the correct number of coordinate-wise multiplications.

    :param args: Experiment arguments. Only max_iter, tol, residue, p, and batch_size are used
    :return: None
    """
    signal = np.array([1.0, 1, 1, 0, 1, 0, 1])
    atoms = np.array(
        [
            np.array([1.0, 0, 1, 0, 1, 0, 1]) / 2,
            np.array([0, 1.0, 0, 1, 0, 1, 0]) / np.sqrt(3),
            np.array([0, 1.0, 0, 0, 0, 0, 0]),
        ]
    )
    coefs, counter = matching_pursuit(signal, atoms, args)
    assert np.allclose(coefs, np.array([2, 0, 1]))
    assert counter == 42

    b_coefs, b_counter = bandit_matching_pursuit(signal, atoms, args)
    assert np.allclose(b_coefs, np.array([2, 0, 1]))
    assert b_counter == 42


def test_case_4(args: argparse.Namespace) -> None:
    """
    Create some random signal and atoms. Verify that Bandit Matching Pursuit returns the same answer as Matching Pursuit.

    NOTE: This function does NOT verify that BMP takes less coordinate-wise computations than MP.

    :param args: Experiment arguments. Unlike the prior test cases, this test case uses args.N, args.d, and args.seed to generate the random data.
    :return: None
    """
    np.random.seed(args.seed)
    data = 2 * np.random.rand(args.N + 1, args.d) - 1  # + 1 to generate the signal
    signal = data[0]
    atoms = data[1:] / np.linalg.norm(data[1:], axis=1).reshape(-1, 1)
    coefs, counter = matching_pursuit(signal, atoms, args)
    print("Test Case 4 MP solution:")
    print(np.where(np.abs(coefs) > 0))
    print(coefs[np.where(np.abs(coefs) > 0)])
    print(counter)
    print()
    print("Test Case 4 BMP solution:")
    b_coefs, b_counter = bandit_matching_pursuit(signal, atoms, args, verbosity=1)
    print(np.where(np.abs(b_coefs) > args.tol))
    print(b_coefs[np.where(np.abs(b_coefs) > args.tol)])
    print(b_counter)
    assert np.allclose(coefs, b_coefs)
    print("Asserted all coefficients are close in Test Case 4")


def main(args: argparse.Namespace) -> None:
    """
    Runs the test cases.

    :param args: Arguments of experiments. Mainly used for test_case_4.
    :return: None
    """
    np.random.seed(args.seed)
    test_case_1(args)
    test_case_2(args)
    test_case_3(args)
    test_case_4(args)


if __name__ == "__main__":
    args = get_args()
    main(args)
    print("All tests pass")
