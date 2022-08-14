import numpy as np
from utils.pca_utils import *
from data.custom_data import generate_custom_data


class PCA_MIPS:
    """
    Basic Class for PCA-MIPS described here:
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf
    """

    def __init__(self, data, query, delta):
        """
        :delta: the number of top principal components to use
        """
        assert data.shape[1] == len(query)

        self.data = data
        self.query = query
        self.delta = delta  # if delta is 0, this is the same as naive MIPS
        assert delta <= len(query)

        self.p_data = None
        self.p_query = None
        self.tree = None

    def preprocess(self):
        """
        Preprocess dataset and signal vector by converting to NN search problem, and building a
        pca-tree data structure. Here, note that the principal components are in "decreasing" order of variance.
        """
        tree = Tree()

        # concatenate data and matmul with principle components matrix
        W, mu, p_data = tree.second_reduction(tree.first_reduction(self.data))
        self.p_data = p_data.T

        # concatenate query and matmul by the same principle components matrix
        _p_query = tree.first_reduction(self.query, is_query=True)
        self.p_query = np.matmul(W.T, _p_query - mu)
        assert self.p_data.shape[1] == len(self.p_query)

        # feed converted data into tree class and construct pca-tree
        tree.data = self.p_data
        self.tree = tree.construct_pca_tree(self.p_data.shape[1] - self.delta)

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
                curr_node = curr_node.left
            else:
                curr_node = curr_node.right
            budget += 1

        # get top k candidates
        candidates = curr_node.idcs
        if len(candidates) < top_k:
            print(f"have {len(candidates)} collisions but need {top_k}!")
            top_k = len(candidates)

        if len(candidates) == 0:
            return budget, []

        dist_vec = np.linalg.norm(self.p_data[candidates] - self.p_query, axis=1)
        idx = np.argpartition(dist_vec, top_k - 1)

        # increment budget by cells seen in matrix multiplication (assuming query is only seen once) and
        # the time complexity of np.argpartition (time complexity is O(n)).
        budget += len(candidates) * len(self.p_query) + len(dist_vec)
        return budget, candidates[idx[:top_k]]


if __name__ == "__main__":
    # This is just for @Mo to test the PR and verify that the lsh-mips object runs.
    # To compare with the naive mips, check branch test_baselines which has a function called compare_to_naive
    delta = 2  # corresponds to the depth of PCA-tree
    top_k = 5

    import time

    atoms, signal = generate_custom_data(num_atoms=100, len_signal=100000)
    signal = signal[0]
    start_time = time.time()
    PCAMIPS = PCA_MIPS(atoms, signal, delta=delta)
    PCAMIPS.preprocess()
    budget, candidates = PCAMIPS.run(top_k=top_k)
    print(time.time() - start_time)
    print(f" =>> Top {top_k} candidates: ", candidates)
    print(" =>> budget: ", budget)
