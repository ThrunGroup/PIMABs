import numpy as np
from utils.pca_utils import *


class PcaMips:
    """
    Basic Class for PCA-MIPS described here:
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf

    :delta: the number of top principal components to use
    """
    def __init__(self, data, query, delta):
        assert (data.shape[1] == len(query))

        self.data = data
        self.query = query
        self.delta = delta
        assert (delta <= len(query))

        self.p_query = None
        self.tree = None

    def preprocess(self):
        """
        Preprocess dataset and signal vector by converting to NN search problem, and building a
        pca-tree data structure. Here, note that the principal components are in "decreasing" order of variance.
        """
        W, mu, p_data = second_reduction(first_reduction(self.data))
        p_data = p_data.T

        _p_query = first_reduction(self.query, is_query=True)
        self.p_query = np.matmul(W.T, _p_query - mu)
        assert (p_data.shape[1] == len(self.p_query))

        self.tree = Tree(p_data, p_data.shape[1] - self.delta)


    def run(self, top_k=1):
        """
        Assumes that dataset and query has already been preprocessed.
        Budget consists of two main operations:
           1. traversing pca-tree
           2. comparing distances
        """
        budget = 0

        # traverse tree to find the query leaf
        curr_node = self.tree.root
        while curr_node.m is not None:
            if self.p_query[curr_node.j] <= curr_node.m:
                curr_node = curr_node.left
            else:
                curr_node = curr_node.right
            budget += 1

        # get top k candidates
        candidates = curr_node.idcs
        if len(candidates) < top_k:
            raise Exception("Don't have k collisions... rerun with different params")

        dist_vec = np.linalg.norm(self.data[candidates] - self.query, axis=1)
        idx = np.argpartition(dist_vec, top_k - 1)

        # increment budget by cells seen in matrix multiplication (assuming query is only seen once) and
        # the time complexity of np.argpartition (time complexity is O(n)).
        budget += (len(candidates) * self.data.shape[1] + len(dist_vec))
        return budget, candidates[idx[:top_k]]
