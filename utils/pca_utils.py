import numpy as np

def first_reduction(arr, is_query=False):
    """
    Apply the first reduction in the preprocessing step and return the matrix with the concatenated rows.
    Conversions are as follows (X is the dataset matrix and y is the query vector):
        x_i -> [sqrt(phi^2 - norm^2);  x_i]
        y -> [0; y]

    :arr: 2d array if is_query is false, otherwise 1d array
    """
    if is_query:
        return np.append(arr, 0)

    norms = np.linalg.norm(arr, axis=1)
    phi = norms.max()
    extra_dim = np.sqrt(phi ** 2 - norms ** 2)
    return np.append(arr, extra_dim.reshape(norms.shape[0], 1), axis=1)


def second_reduction(arr):
    # Todo: check if these dimensions are correct!!!
    """
    Apply the second reduction in the pre-processing step and return the transformed arr.
    Note that the components are in decreasing order of variance.   #Todo: verify that this is true!!!
    Converts X -> W^T (X - mu) where
        - mu: 1/n * sum(X_i)
        - (X - M) = W Sig U^T where M is mu concatenated as columns
    """
    mu = np.mean(arr, axis=1)
    M = np.array([])
    for _ in range(arr.shape[1]):
        M = np.c_[M, mu]

    W, _, _ = np.linalg.svd(arr - M, full_matrices=True)
    return np.matmul(W.T, arr - mu)


class Tree:
    def __init__(self, data, j):
        self.data = data
        self.root = self.construct_pca_tree(j)

    def pca_tree_rec(self, node, j, max_j):
        if j == max_j:
            return

        # find median value of node data
        node_data = self.data[node.idcs]
        m = np.median(node_data[:, j])
        node.m = m
        node.j = j

        # initialize children
        l_idcs = node.idcs[np.where(node_data[:, j] <= m)]
        r_idcs = node.idcs[np.where(node_data[:, j] > m)]
        node.left = Node(l_idcs)
        node.right = Node(r_idcs)

        pca_tree_rec(node.left, j + 1, max_j)
        pca_tree_rec(node.right, j + 1, max_j)
        return

    def construct_pca_tree(self, j):
        """
        :delta: number of top principal components to use.
        """
        N, F = self.data.shape()
        root = Node(np.arange(N))
        pca_tree_rec(root, j, F)
        return root


class Node:
    def __init__(self, idcs):
        self.idcs = idcs
        self.m = None
        self.j = None

        self.left = None
        self.right = None
