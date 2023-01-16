import numpy as np
import numpy.linalg as la
#from algorithms.matching_pursuit import orthogonal_matching_pursuit
from utils.constants import ACTION_ELIMINATION


def create_toy_discrete(num_atoms, mul):
    """
    This function creates a dictionary of atoms where the first three atoms are unit atoms a, b, c, and the rest of the
    atoms are zero vectors. The query vector is q = 3a + 2b + c with the objective of MP running three iterations
    to find the indices 0, 1, 2.
    """
    assert mul % 2 == 0 and mul >=2, "mul must be a multiple of 2 and greater than 2"
    a = np.array([1, 1, 1] * mul)
    b = np.array([-1, 0, 1] * mul)
    c = np.concatenate((np.array([1, 0, 1] * (mul//2)), np.array([-1, 0, -1] * (mul//2))))
    random_arr = np.random.rand(num_atoms-3, len(a))
    atoms = np.vstack(
        (a/la.norm(a), b/la.norm(b), c/la.norm(c), random_arr/la.norm(random_arr, axis=1)[np.newaxis])
    )
    query = 3*a + 2*b + c
    return atoms, query


def create_toy_waves(num_atoms, d):
    """
    This funtion creates a dictionary of atoms of sinusoid waves where the first three sine waves a, b, c are orthogonal
    to each other with the same amplitude and the rest of the atoms are zero vectors.
    The query vector is q = 3a + 2b + c.
    """
    amplitude = 2
    t = np.linspace(0, 2 * np.pi * d, d)
    # generate 90 degree offset waves
    a = amplitude * np.sin(t)
    b = amplitude * np.sin(2 * t)
    c = amplitude * np.sin(3 * t)

    random_arr = np.random.uniform(low=-amplitude, high=amplitude, size=(num_atoms-3, d))
    atoms = np.vstack(
        (a/la.norm(a), b/la.norm(b), c/la.norm(c), random_arr/la.norm(random_arr, axis=1)[:, np.newaxis])
    )
    query = 3*a + 2*b + c
    return atoms, query


def run_toy_dataset(
        name="discrete",
        num_atoms=100,
        mul=3000
):
    """
    Runs bandit-OMP on the specified toy dataset. The objective is for the OMP to find the first, second, third
    atom indices, in that order
    """
    print(f"number of atoms: {num_atoms}")
    print(f"dimension of query: {mul * 3}")
    if name == "discrete":
        print("\n-------- these are the results for the toy discrete --------")
        atoms, query = create_toy_discrete(num_atoms, mul)
    else:
        print("\n-------- these are the results for the sine toy --------")
        atoms, query = create_toy_waves(num_atoms, mul * 3)

    candidates, budget = orthogonal_matching_pursuit(
        atoms=atoms.T,
        signal=query,
        n_nonzero_coefs=3,
        min_residual=0.01,
        bandit_alg=ACTION_ELIMINATION,
        abs=True,
        seed=0,
        use_cache=False,
    )
    print("indices: ", candidates)


if __name__ == "__main__":
    run_toy_dataset(name="discrete")
    run_toy_dataset(name="waves")