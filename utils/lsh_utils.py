import numpy as np
import math


def binary_to_num(bnum):
    """
    Converts an array of ints into one integer.
    [1, 0, 1, 1] => 1011
    """
    return int("".join(map(str, bnum)))


def g_index_extend(data, m):
    """
    [x] => [x;    ||x||**2; ||x||**4; ...; ||x||**(2*m)]
    """
    result = []
    for row in data:
        result.append(
            (row + [np.dot(row, row) ** (i + 1) for i in range(m)])
        )
    return result


def g_query_extend(query, m):
    """
    [x] => [x;    1/2; ...; 1/2]
    """
    return query + [0.5 for i in range(m)]


def g_index_simple_extend(data):
    """
    This should only be called when we're using symmetric hash functions.
    [x] => [x;   sqrt(1 - ||x||**2)]
    """
    result = []
    for row in data:
        result.append(
            (row + [np.sqrt(1 - np.dot(row, row))])
        )
    return result


def g_query_simple_extend(query):
    """
    [x] => [x;   sqrt(1 - ||x||**2)]
    """
    assert (round(np.dot(query, query)) == 1)
    return query + [0]


def g_transformation(data):
    """
    A scaling transformation S(.) to convert MIPS to NN problem
    Conversion is as follows (U < 1  ||xi||_2 <= U <= 1):
        S(x) = x * U/M
    where M = max_i ||x_i||_2
    """
    U = 0.83
    norm_list = [math.sqrt(np.dot(dd, dd)) for dd in data]
    max_norm = max(norm_list)
    ratio = float(U / max_norm)
    return ratio, max_norm, [[ratio * dx for dx in dd] for dd in data]


def g_normalization(query):
    """
    Normalize the query vector so that ||q||_2 = 1
    """
    if int(np.sum(query == 0)):
        len = len(query)
        return [q / len for q in query]

    norm = math.sqrt(np.dot(query, query))
    return [q / norm for q in query]


if __name__ == "__main__":
    seed = 0
    atoms, signal = generate_normal_data(num_atoms=100, len_signal=1000, seed=seed)
    run_lsh_mips(atoms, signal, type=SYMMETRIC, verbose=True)
    run_lsh_mips(atoms, signal, type=ASYMMETRIC, verbose=True)