import numpy as np
import numba as nb


def subset_2d_cached(source_array, rows, columns, cache, cache_tracker, cache_map):
    """
    Implements subset_2d with cache support. Assumes budget to read/write cache is the same.
    Returns the subset_array, updated cache, cache_tracker, and budget.
    """
    budget = 0
    subset_array = np.empty((len(rows), len(columns)))
    print("----------------------------------")
    print(f"rows: {rows}\n cols: {columns}")
    for i in range(len(rows)):
        for j in range(len(columns)):
            atom = rows[i]
            sample = columns[j]

            # Check if sample already exists in cache.
            if cache_map[atom].get(sample, -1) >= 0:
                subset_array[(i, j)] = cache[(atom, cache_map[atom][sample])]
                budget += 0.01

            else:
                # write to cache if space exists
                source = source_array[(atom, sample)]
                if (cache_tracker[atom] < cache.shape[1]):
                    cache[(atom, cache_tracker[atom])] = source
                    cache_map[atom][sample] = cache_tracker[atom]
                    cache_tracker[atom] += 1
                    budget += 0.01

                subset_array[(i, j)] = source
                budget += 1
    return subset_array, budget, cache, cache_tracker, cache_map


def print_datastructures(subset_array, cache, cache_tracker, cache_map):
    print(f"subset_array: \n{subset_array}\n")
    print(f"cache: \n{cache}\n")
    print(f"tracker: \n{cache_tracker}\n")
    print(f"cache_map (cols -> cache_idx): \n{cache_map}\n")


if __name__ == "__main__":
    source_array = np.arange(30).reshape((3, 10))
    cache = np.zeros((3, 4)) - np.ones((3, 4))
    cache_tracker = np.zeros(3, dtype=np.int64)  # each element < 4 since this is cache size
    cache_map = nb.typed.List()
    for i in range(3):
        cache_map.append(nb.typed.Dict.empty(key_type=nb.int64, value_type=nb.int64))

    str_list = ["write", "read", "write", "read + sample", "write", "write + sample + sample"]
    rows_list = [[0, 2], [0, 2], [0 ,2], [0, 2], [1], [0,1,2]]
    cols_list = [[0, 2], [0, 2], [3, 4], [4, 5], [6], [7]]
    print(f"original array: \n {source_array}\n")
    print("----------------------------------")
    print_datastructures([], cache, cache_tracker, cache_map)

    inputs = zip(rows_list, cols_list)
    for i, input in enumerate(inputs):
        subset_array, budget, cache, cache_tracker, cache_map = subset_2d_cached(
            source_array, input[0], input[1], cache, cache_tracker, cache_map
        )
        print(f"-> {str_list[i]} with budget {budget}\n")
        print_datastructures(subset_array, cache, cache_tracker, cache_map)