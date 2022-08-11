import numpy as np
from utils.pca_utils import Tree, Node

def test_tree(data):
    tree = Tree()
    tree.data = data
    root = tree.construct_pca_tree(j=3)
    tree.print_tree_rec(root, " ")


def test_pca(data):


if __name__ == "__main__":
    data = np.random.randint(0, 5, size=20).reshape((4, 5))
    print(data)
    test_tree(data)