from multiprocessing import cpu_count
import numpy as np
import numba as nb
import math


def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1)
    norms_matrix = norms[:, np.newaxis]
    normalized_vecs = np.divide(vecs, norms_matrix, out=np.zeros_like(vecs), where=norms_matrix != 0)  # divide by zero problem
    return norms, normalized_vecs


@nb.jit
def true_positives(topK, Q, G, T):
    result = np.empty(shape=(len(Q)))
    for i in nb.prange(len(Q)):
        result[i] = len(np.intersect1d(G[i], topK[i][:T]))
    return result


@nb.jit
def parallel_sort(compressed, Q):
    """
    For each q in 'Q', sort the items in "compressed" by their euclidean product distance.
    :param compressed: compressed items, same dimension as origin data, shape(N * D)
    :param Q: queries, shape(len(Q) * D)
    :return: indices of the topk closest items and the budget
    """
    budget = 0
    rank = np.empty((Q.shape[0], min(131072, compressed.shape[0]-1)), dtype=np.int32)
    for i in nb.prange(Q.shape[0]):
        distances = compressed @ -Q[i]
        budget += len(Q[i])

        # originally the method is to argpartition -> argsort, but numba doesn't support argpartition
        top_k = min(131072, len(distances) - 1)
        rank[i, :] = np.argsort(distances)[:top_k]
        budget += (len(distances) + top_k * math.log(top_k))
    return rank, budget


class Sorter(object):
    def __init__(self, compressed, Q):
        self.Q = Q
        self.topK, budget = parallel_sort(compressed, Q)
        self.budget = budget

    def recall(self, G, T):
        t = min(T, len(self.topK[0]))
        return t, self.sum_recall(G, T) / len(self.Q)

    def sum_recall(self, G, T):
        assert len(self.Q) == len(self.topK), "number of query not equals"
        assert len(self.topK) <= len(G), "number of queries should not exceed the number of queries in ground truth"
        true_positive = true_positives(self.topK, self.Q, G, T)
        return np.sum(true_positive) / len(G[0])  # TP / K


class BatchSorter(object):
    def __init__(self, compressed, Q, G, Ts, batch_size):
        self.Q = Q
        self.recalls = np.zeros(shape=(len(Ts)))
        self.budget = 0

        for i in range(math.ceil(len(Q) / float(batch_size))):
            q = Q[i*batch_size: (i + 1) * batch_size, :]
            g = G[i*batch_size: (i + 1) * batch_size, :]
            sorter = Sorter(compressed, q)
            self.budget += sorter.budget
            self.recalls[:] = self.recalls[:] + [sorter.sum_recall(g, t) for t in Ts]

        self.recalls = self.recalls / len(self.Q)

    def recall(self):
        return self.recalls, self.budget
