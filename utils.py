import os
import argparse
import numpy as np
from typing import List


def get_args() -> argparse.Namespace:
    """
    Gather the experimental arguments from the command line.

    :return: Namespace object from argparse containing each argument
    """
    parser = argparse.ArgumentParser(description="Solve matching pursuit problems")
    parser.add_argument("-N", help="Number of atoms to use", default=100, type=int)
    parser.add_argument("-d", help="Dimensionality of data", default=1000, type=int)
    parser.add_argument(
        "--seed", help="Random seed for experiments", default=1, type=int
    )
    parser.add_argument(
        "--p",
        help="Confidence parameter that governs the width of the confidence intervals",
        type=float,
    )
    parser.add_argument(
        "--batch_size",
        help="Number of coordinates to sample at once",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "--max_iter", help="Maximum number of atoms to identify", default=10, type=int
    )
    parser.add_argument(
        "--tol",
        help="Minimum absolute value of inner product between atom and signal",
        default=1e-2,
        type=float,
    )
    parser.add_argument(
        "--residue",
        help="Threshold for norm of residue by which to terminate iterative algorithm.",
        default=1e-3,
        type=float,
    )
    parser.add_argument(
        "-e",
        "--exp_config",
        help="Experiment configuration file to use",
        required=False,
    )
    parser.add_argument(
        "-f", "--force", help="Recompute Experiments", action="store_true"
    )
    return parser.parse_args()


def remap_args(args: argparse.Namespace, exp: List):
    """
    Parses an experiment config line (as a list) into an args variable (a Namespace).

    :param args: Namespace object whose args are to be modified
    :param exp: Experiment configuration (list of arguments)
    """
    args.N = exp[1]
    args.d = exp[2]
    args.seed = exp[3]
    args.p = None  # To compute on the fly
    args.batch_size = exp[5]
    args.max_iter = exp[6]
    args.tol = exp[7]
    args.residue = exp[8]
    return args


def q(str_: str) -> str:
    """
    Helper function that prepends and appends double quotes to a string.

    :param str_: string to put double quotes around
    """
    return '"' + str_ + '"'


def write_exp(algo, N, d, seed, p, batch_size, max_iter, tol, residue) -> str:
    """
    Given a set of experimental arguments, produce a string representing the list of arguments (to be stored in a text
    file)

    :param algo: algorithm to use, e.g., MP BanditMP
    :param N: dataset size
    :param d: dimensionality
    :param seed: random seed
    :param p: confidence parameter
    :param batch_size: batch size to use when sampling
    :param max_iter: maximum number of iterations
    :param tol: tolerance
    :param residue: residue minimum
    :return: string representation of list of arguments
    """
    return (
        "\t["
        + ", ".join(
            map(str, [q(algo), N, d, seed, p, batch_size, max_iter, tol, residue])
        )
        + "],\n"
    )


def conv_to_str(input_: List[int]) -> str:
    """
    Convenience function for printing an array of numbers to the same number of decimal places. Return a string
    representation of the given input array for easy printing.
    :param input_: input array to convert to a string
    :return: string representation of the array
    """
    return ", ".join(map(str, map(lambda num: np.round(num, decimals=5), input_)))


def write_log(
    args: argparse.Namespace,
    stats: List,  # Actually a List[List[np.ndarray, int, int, float]]
    coefs: np.ndarray,
    init_signal_norm: float,
    final_signal_norm: float,
    total_cost: int,
    bandit: bool,
):
    """
    Writes a logfile containing information about the experiment. The logfile contains information that is very useful
    for debugging, such as exactly which atoms were chosen at each step and the total cost of the algorithm.

    :param args: arguments about the experiment.
    :param stats: statistics about the experiments
    :param coefs: coefficients (inner products) with each atom at each iteration of MP
    :param init_signal_norm: initial norm of the input signal
    :param final_signal_norm: final norm of the residual (should be very small) - useful for debugging
    :param total_cost: cost of the experiment, measured in number of coordinate-wise multiplications performed
    :param bandit: whether the experiment used BanditMP (True) or MP (False)
    :return: None
    """
    nonzero_coefs = list(np.where(np.abs(coefs) > 0)[0])
    nonzero_weights = list(coefs[nonzero_coefs])
    with open(get_logfile_name(args, bandit), "w+") as fout:
        fout.write("Total cost: " + str(total_cost) + "\n")
        fout.write("Final answer:\n")
        fout.write("Nonzero indcs: " + conv_to_str(nonzero_coefs) + "\n")
        fout.write("Nonzero coefs: " + conv_to_str(nonzero_weights) + "\n")
        fout.write("Initial signal norm: " + str(init_signal_norm) + "\n")
        for row_i, row in enumerate(stats):
            fout.write("Iteration: " + str(row_i + 1) + "\n")
            fout.write(conv_to_str(row))
            fout.write("\n")
        fout.write("Final signal norm: " + str(final_signal_norm) + "\n")


def get_logfile_name(args: argparse.Namespace, bandit=True) -> str:
    """
    Returns the name of the logfile with the given arguments.

    :param args: arguments of the experiment
    :param bandit: whether to use BanditMP (True) or MP (False)
    :return: string representing the logfile's name
    """
    suffix = "-bmp" if bandit else ""
    return os.path.join(
        "logs",
        "N-"
        + str(args.N)
        + "-d-"
        + str(args.d)
        + "-s-"
        + str(args.seed)
        + "-p-"
        + str(args.p)
        + "-b-"
        + str(args.batch_size)
        + "-m-"
        + str(args.max_iter)
        + "-r-"
        + str(args.residue)
        + "-t-"
        + str(args.tol)
        + suffix,
    )
