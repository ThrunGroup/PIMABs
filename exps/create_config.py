"""
Convenience code to automatically generate a list of experiments to run.
Default outpu file is  auto_exp_config.py.
"""

import itertools
import sys

sys.path.append("..")
from utils import write_exp


def main() -> None:
    """
    Write the experiments you want to run to a file that will be ingested by run.py.

    Modify the arrays below to modify the generated experiments.

    :return: None
    """
    # Algorithms to run: Bandit Matching Pursuit (B_MP) and/or Matching Pursuit (MP)
    algos = ["B_MP"]  # ], "MP"]

    # Number of atoms to use
    Ns = [783]

    # Dimensionalities of vectors to use
    ds = [
        100,
        200,
        400,
        800,
        1600,
        3200,
        6400,
        12800,
        25600,
        51200,
    ]

    # Seeds to use
    seeds = range(10)

    # Maximum number of matching pursuit iterations (iterations of MIPS)
    max_iters = [1]

    # Tolerance levels to use
    tols = [1e-2]

    # Maximum residual values to use
    residues = [1e-2]

    # Placeholder variables to generate the args
    ps = [1e-2]  # This is unused by the experiments and recomputed in-method
    batch_sizes = [10]  # This is unused by the experiments and recomputed in-method

    exps = itertools.product(
        algos, Ns, ds, seeds, ps, batch_sizes, max_iters, tols, residues
    )
    with open("auto_exp_config.py", "w+") as fout:
        fout.write("experiments = [\n")
        for e in exps:
            exp = write_exp(*e)
            fout.write(exp)
        fout.write("]")


if __name__ == "__main__":
    main()
