import random
import numpy as np
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from operator import itemgetter

from ..utils.lsh_utils import *

class LshMips:
    """
    Class for solving MIPS using LSH. Implements symmetric LSH (See https://arxiv.org/abs/1410.5518)
    and asymmetric LSH (https://dl.acm.org/doi/10.5555/2969033.2969086) by referring to the
    implementation in https://github.com/jfpu/lsh_mips.

    :type: either symmetric or asymmetric
    :data: data used to build hash indices
    :queries: 1d array of query data
    :rand_range: random range for normalization
    :num_neighbors: number of neighbors to find in the near neighbor search.
                    If this is 1, we're finding the nearest neighbor.
    """

    def __init__(self, type, data, query, rand_range=1.0, m=3):
        kdata = len(data[0])
        qdata = len(query)    # queries are 1d-array
        assert (kdata == qdata)     # can only take inner product if dimensions match

        self.type = type
        self.data = data
        self.query = query

        self.rand_range = rand_range
        self.num_neighbors = num_neighbors

        # m is always 1 if we're doing symmetric lsh mips
        if type == "symmetric":
            self.m = 1  # query gets extended the same way as dataset
        else:
            self.m = m
            self.half_extend = g_ext_half(self.m)

        # data & query transformation. norm_data is S(x) in the original paper
        _, _, self.norm_data = g_transformation(data)
        self.norm_query = g_normalization(query)
        assert (kdata == len(self.norm_data[0]))
        assert (qdata == len(self.norm_query))

        # preprocessing: expand k dimension into k+2m dimension following directions in original paper
        new_len = kdata + self.m
        self.d = new_len
        if type == "symmetric":
            self.ext_data = g_index_simple_extend(self.norm_data, self.m)
            self.ext_query = g_query_simple_extend(self.norm_query, self.m)
        else:
            self.ext_datas = g_index_extend(self.norm_datas, self.m)
            self.ext_query = g_query_extend(self.norm_query, self.m)

        assert (new_len == len(self.ext_data[0]))   # num dimensions
        assert (new_len == len(self.ext_query))

    def run(self, type, num_tables=2, num_hash_functions=2, top_k=1):
        """
        Run the nearest neighbor search to find the specified number of neighbors (default 1).

        :num_tables: number of hash tables
        :num_hfunctions: number of hash functions per table (i.e. size of the hash table)
        """
        lsh = LshWrapper(type, self.d, self.rand_range, num_tables, 0)
        lsh.resize(num_hash_functions)
        lsh.index(self.ext_data)    # hash the extended data (i.e. P(x))
        budget, candidates = lsh.query(self.ext_query, top_k)

        # find best candidate and increment budget
        dist_vec = np.linalg.norm(self.data[candidates] - self.query, axis=1)
        idx = np.argpartition(dist_vec, top_k)

        # increment budget by cells seen in matrix multiplication (assuming query is only seen once) and
        # the time complexity of np.argpartition (time complexity is O(n)).
        budget += (len(candidates) * self.data.shape[1] + len(dist_vec))
        return budget, candidates[idx[:top_k]]

