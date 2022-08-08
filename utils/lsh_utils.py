import math
import random
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from operator import itemgetter
from global_mgr import gol

import numpy as np
from lsh import *

class Hash:
    'hash base'

    __metaclass__ = ABCMeta

    def __init__(self, d):
        self.d = d
        self.hyperplanes = [random.gauss(0, 1) for i in xrange(self.d)]

    @abstractmethod
    def hash(self, vec):
        pass


class SymmetricLsh(Hash):
    """
    Hash Base class
    """
    def __init__(self, d):
        Hash.__init__(self, d)

    def hash(self, vec):
        """
        Hash function defined in the papr https://arxiv.org/abs/1410.5518.
        h(x) = sign(a @ P(x))
        """
        return len(vec), int(0 < dot(vec, self.hyperplanes))



class AsymmetricLsh(Hash):
    # r: fixed size
    # d: data length
    # RandomData: random data vector
    def __init__(self, d):
        Hash.__init__(self, d)
        self.r = 1.0    # Todo: should this be 1.0?
        self.b = random.uniform(0, self.r)  # 0 < b < r

    def hash(self, vec):
        return len(vec), int((dot(vec, self.hyperplanes) + self.b) / self.r)


class LshWrapper:
    # lsh_type: lsh hash func type: in 'l2' 'cosine'
    # r: random float data
    # d: data vector size
    # k: number of hash func for each hashtable. default 2
    # L: number of hash tables for each hash type: default 2
    def __init__(self, type, d, r=1.0, k=2, L=2):
        self.d = d
        self.r = r
        self.k = k
        self.L = 0

        if type == "symmetric":
            self.hash_class = SymmetricLsh
        else:
            self.hash_class = AsymmetricLsh

        self.hash_tables = []
        self.resize(L)

    def resize(self, L):
        # shrink the number of hash tables to be used
        if L < self.L:
            self.hash_tables = self.hash_tables[:L]
        else:
            # initialise a new hash table for each hash function
            hash_classes = []
            for i in xrange(self.L, L):
                hash_classes.extend(
                    # each hash_class will have different hyperplanes
                    [self.hash_class(self.d) for _ in xrange(self.k)]
                )
            # create a table for each of the hash_funcs (i.e. hash classes)
            # single element of table looks like (hash_class, dict(hash_value: {idcs_of_data}))
            self.hash_tables.extend(
                [(g, defaultdict(lambda:[])) for g in hash_classes]
            )

        self.L = L

    def index(self, data):
        """
        Hash the dataset (this should be the extended dataset) with the hash functions in the hash table.
        If we have L hash functions, the entire dataset will be hashed L times.
        """
        self.data = data
        for hash_class, hash_dict in self.hash_tables:
            for i, row in enumerate(self.data):
                hash_dict[hash_class.hash(row)].append(i)

    def query(self, q, top_k=1):
        """
        Return the best candidate and budget of MIPS. Here, budget is the number of cells that were used in MIPS where
        cells equals x_{ij} (i.e. a single element in a vector). This function returns the row ind(ex/ices) of the best
        candidate.
        """
        budget = 0
        candidates = set()      # this prevents duplicate indices
        for hash_class, hash_dict in self.hash_tables:
            hist_budget, q_hash = hash_class.hash(q)
            matches = hash_dict.get(q_hash, [])     # return [] if match doesn't exist
            candidates.update(matches)
            budget += hist_budget
        return budget, candidates


# matrix inner product
def dot(u, v):
    # return sum(ux * vx for ux, vx in zip(u,v))
    return np.dot(u, v)


def g_ext_norm(vec, m):
    l2norm_square = dot(vec, vec)
    return [l2norm_square**(i+1) for i in xrange(m)]


def g_ext_half(m):
    return [0.5 for i in xrange(m)]


def g_index_extend(data, m):
    """
    [x] => [x;    ||x||**2; ||x||**4; ...; ||x||**(2*m)]
    """
    return [(dv + g_ext_norm(dv, m)) for dv in data]


def g_query_extend(queries, m):
    """
    [x] => [x;    1/2; ...; 1/2]
    """
    return [(qv + g_ext_half(m)) for qv in queries]


def g_ext_zero(m):
    return [0 for i in xrange(m)]


def g_ext_norm_simple(vec, m):
    l2norm_square = dot(vec, vec)
    return [np.sqrt(1 - l2norm_square) for i in xrange(m)]


def g_index_simple_extend(data, m):
    """
    This should only be called when we're using symmetric hash functions.
    [x] => [x;    sqrt(1 - ||x||**2);    0]
    """
    assert (1 == m)
    return [(dv + g_ext_norm_simple(dv, m)) for dv in data]


def g_query_simple_extend(query, m):
    """
    [x] => [x;    0;    sqrt(1 - ||x||**2)]
    """
    assert (1 == m)
    return [(query + g_ext_norm_simple(query, m))]


# get max norm for two-dimension list
def g_max_norm(data):
    norm_list = [math.sqrt(dot(dd, dd)) for dd in data]
    return max(norm_list)


# data transformation. S(xi) = (U / M) * xi
def g_transformation(data):
    # U < 1  ||xi||2 <= U <= 1. recommend 0.83
    U = 0.83
    max_norm = g_max_norm(data)
    ratio = float(U / max_norm)
    return ratio, max_norm, [[ratio * dx for dx in dd] for dd in data]


# normalization for each query
def g_normalization(query):
    # Todo: Should U be set to 1??
    U = 0.83
    norm = math.sqrt(dot(qv, qv))
    ratio = float(U / norm)
    return [ratio * i for i in query]
