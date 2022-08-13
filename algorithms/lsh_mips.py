import random
import numpy as np
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from operator import itemgetter

from utils.lsh_utils import *
from utils.constants import SYMMETRIC, ASYMMETRIC

# for debugging purposes
np.random.seed(0)

class LSH_MIPS:
    """
    Class for solving MIPS using LSH. The main logic divides into preprocessing the data and
    querying the signal vector. Used https://github.com/jfpu/lsh_mips as reference for implementation.
    The basic logic of running LSH_MIPS is as follows:
        1. initialize LSH_MIPS class with all it's parameters
        2. call prepare_data() method and create_lsh_table()
        3. call run()

    :type: either symmetric or asymmetric
    :data: data used to build hash indices
    :queries: 1d array of query data
    :rand_range: random range for normalization
    :m: the number of dimensions to extend the dataset to convert from MIPS to NN search problem.
    """

    def __init__(
        self,
        data,
        query,
        m=3,
        rand_range=1.0,
        num_hfunc=5,
        num_tables=10,
        type=ASYMMETRIC,
    ):
        kdata = len(data[0])
        qdata = len(query)  # queries are 1d-array
        assert (kdata == qdata)  # can only take inner product if dimensions match

        self.data = data
        self.query = query

        if type == SYMMETRIC:
            self.m = 1  # m is always 1 if we're doing symmetric lsh mips
            self.hash_func = lambda data, hyperplanes: \
                                np.matmul(data, hyperplanes)

        elif type == ASYMMETRIC:
            self.m = m
            b = np.random.uniform(0, rand_range)  # 0 < b < rand_range
            self.hash_func = lambda data, hyperplanes: \
                                (np.matmul(data, hyperplanes) + b) / rand_range
        else:
            raise Expection("Invalid LSH type declared")

        self.type = type
        self.num_hfunc = num_hfunc
        self.num_tables = num_tables

    def prepare_data(self):
        """
        Prepare the data before hashing into buckets. Preprocessing includes these steps:
            1. scaling transformation of data (to convert from MIPS to NNS)
            2. vector concatenation corresponding to P(.) and Q(.) of the data (symmetric lsh only has P(.))
        """
        # data & query transformation. norm_data is S(x) in the original paper
        _, _, norm_data = g_transformation(self.data)
        norm_query = g_normalization(self.query)
        self.d = len(norm_data[0]) + self.m

        # expand k dimension into k+2m dimension following directions in original paper
        if self.type == SYMMETRIC:
            self.ext_data = np.array(g_index_simple_extend(norm_data))
            self.ext_query = np.array(g_query_simple_extend(norm_query))
        else:
            self.ext_data = np.array(g_index_extend(norm_data, self.m))
            self.ext_query = np.array(g_query_extend(norm_query, self.m))

        assert (self.d == len(self.ext_data[0]))   # num dimensions
        assert (self.d == len(self.ext_query))

    def create_lsh_table(self, data):
        """
        Hash the dataset into it's corresponding buckets. Each element in the bucket is the "index" of the dataset row.

        :data: the preprocessed dataset
        """
        assert (data.shape[1] == self.d)  # make sure that we're using the preprocessed data

        # create num_tables hash tables where each table consists of num_hfunc hyperplanes
        self.hash_tables = []
        for i in range(self.num_tables):
            if self.num_hfunc == 0:
                hyperplanes = np.zeros((1, self.d))  # map to single bucket
            else:
                hyperplanes = np.random.normal(0.0, 1.0, (self.num_hfunc, self.d))

            self.hash_tables.append(
                # single element of table looks like (hyperplanes, dict(hash_value: {idcs_of_data}))
                (hyperplanes, defaultdict(lambda: []))
            )

        # hash the values
        for hyperplanes, hash_dict in self.hash_tables:
            # standard amplification procedure (a hash value is an array).
            # The final number of buckets is 2^num_hfunc
            values = self.hash_func(data, hyperplanes.T)
            hash_values = np.where(values > 0, 1, 0)

            for i, hash_value in enumerate(hash_values):
                bucket = binary_to_num(hash_value)
                hash_dict[bucket].append(i)

    def run(self, data, query, top_k=1):
        """
        Run the nearest neighbor search to find the specified number of neighbors (default 1).

        :data: the preprocessed data. Only used when finding the best match out of collisions.
        :query: the preprocessed query to run NNS on
        :top_k: the number of top candidates you want to find
        """
        # make sure we're using the preprocessed data and queries
        assert(data.shape[1] == self.d)
        assert(len(query) == self.d)

        budget = 0
        candidates = set()  # this prevents duplicates
        for hyperplanes, hash_dict in self.hash_tables:
            value = self.hash_func(hyperplanes, query)
            hash_value = np.where(value > 0, 1, 0)

            # get matches
            bucket = binary_to_num(hash_value)
            matches = hash_dict.get(bucket, [])  # return [] if match doesn't exist
            candidates.update(matches)

            # dot product with one hyperplane costs self.d budget
            h, d = hyperplanes.shape
            budget += h * d

        # return budgets and candidates matched
        candidates = np.asarray(list(candidates))

        # There aren't enough candidates. Still return the ones that we have
        if len(candidates) < top_k:
            print(f"have {len(candidates)} collisions but want {top_k}!")
            top_k = len(candidates)

            if top_k == 0:
                return budget, []

        # NN search for the best top_k candidates. Need to check all candidates at least once to compare with query.
        # and one more time since the complexity of np.argpartition is O(n) (uses a heap).
        dist_vec = np.linalg.norm(data[candidates] - query, axis=1)
        idx = np.argpartition(dist_vec, top_k - 1)
        budget += (len(candidates) + 1) * self.d
        return budget, candidates[idx[:top_k]]

