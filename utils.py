import os
import argparse
from typing import List


def get_args() -> argparse.Namespace:
    """
    Gather the experimental arguments from the command line.

    :return: Namespace object from argparse containing each argument
    """
    parser = argparse.ArgumentParser(description="Solve matching pursuit problems")
    parser.add_argument("-N", default=100, type=int)
    parser.add_argument("-d", default=1000, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--p", type=float)
    parser.add_argument("--batch_size", default=1000, type=int)
    parser.add_argument("--max_iter", default=10, type=int)
    parser.add_argument("--tol", default=1e-2, type=float)
    parser.add_argument("--residue", default=1e-3, type=float)
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
