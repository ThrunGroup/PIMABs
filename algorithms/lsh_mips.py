import random
import numpy as np
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from operator import itemgetter

from utils.lsh_utils import *

class LshMips:
    """
    Class for solving MIPS using LSH. The main logic divides into preprocessing the data and
    querying the signal vector. Used https://github.com/jfpu/lsh_mips as reference for implementation.

    :type: either symmetric or asymmetric
    :data: data used to build hash indices
    :queries: 1d array of query data
    :rand_range: random range for normalization
    :m: the number of dimensions to extend the dataset to convert from MIPS to NN search problem.
    """

    def __init__(self, type, data, query, rand_range=1.0, m=3):
        kdata = len(data[0])
        qdata = len(query)    # queries are 1d-array
        assert (kdata == qdata)     # can only take inner product if dimensions match

        self.type = type
        self.data = data
        self.query = query
        self.rand_range = rand_range

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
            self.ext_data = g_index_extend(self.norm_data, self.m)
            self.ext_query = g_query_extend(self.norm_query, self.m)

        assert (new_len == len(self.ext_data[0]))   # num dimensions
        assert (new_len == len(self.ext_query))

    def run(self, type, num_hfunc=2, num_tables=2, top_k=1):
        """
        Run the nearest neighbor search to find the specified number of neighbors (default 1).

        :num_tables: number of hash tables
        :num_hfunctions: number of hash functions per table (i.e. size of the hash table)
        """
        lsh = LshWrapper(type, d=self.d, r=self.rand_range, num_hfunc=num_hfunc, num_tables=num_tables)
        lsh.resize(num_tables)
        lsh.index(self.ext_data)    # hash the extended data (i.e. P(x))
        budget, candidates = lsh.query(self.ext_query, top_k)

        candidates = np.array(list(candidates))
        if len(candidates) < top_k:
            raise Exception("Don't have k collisions... rerun with different params")

        # find best candidate and increment budget
        dist_vec = np.linalg.norm(self.data[candidates] - self.query, axis=1)
        idx = np.argpartition(dist_vec, top_k - 1)

        # increment budget by cells seen in matrix multiplication (assuming query is only seen once) and
        # the time complexity of np.argpartition (time complexity is O(n)).
        budget += (len(candidates) * self.data.shape[1] + len(dist_vec))
        return budget, candidates[idx[:top_k]]

