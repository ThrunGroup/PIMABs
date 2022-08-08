import numpy as np
from ..utils.pca_utils import *


class PcaMips:
    def __init__(self, data, query, delta):
        self.data = data
        self.query = query
        self.delta = delta

        self.p_query = None
        self.tree = None

    def preprocess(self):
        p_data = second_reduction(first_reduction(self.data))
        self.p_query = second_reduction(first_reduction(self.query, is_query=True))
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
        curr_node = self.tree
        while curr_node.m is not None:
            if self.p_query[curr_node.j] <= curr_node.m:
                curr_node = self.tree.left
            else:
                curr_node = self.tree.right
            budget += 1

        # get top k candidates
        candidates = curr_node.idcs
        dist_vec = np.linalg.norm(self.data[candidates] - self.query, axis=1)
        idx = np.argpartition(dist_vec, top_k)

        # increment budget by cells seen in matrix multiplication (assuming query is only seen once) and
        # the time complexity of np.argpartition (time complexity is O(n)).
        budget += (len(candidates) * self.data.shape[1] + len(dist_vec))
        return budget, candidates[idx[:top_k]]
