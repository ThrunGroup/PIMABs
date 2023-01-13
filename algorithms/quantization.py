"""
This entire codebase was adopted from the github of the authors of the "Norm-Explicit Quantization" paper from the
link https://github.com/xinyandai/product-quantization/tree/eaa7aa362eb7cc9392e34561644561c9b341c221.
"""

from __future__ import division
from __future__ import print_function
import numpy as np
import math
from scipy.cluster.vq import vq, kmeans2
from utils.quantization_sorters import normalize, BatchSorter

from data.get_data import get_data
from utils.constants import (
    NETFLIX,
    NORM_BUFFER
)


class NormPQ(object):
    """
    Receives ResidualPQ object since that it what's going to be used to train the direction and norm codebooks.
    The member functions are as follows:
        - fit: trains the same number of direction and norm codebooks following Algorithm 2 from the paper
        - {encode/decode}_norm: encodes/decodes the norms with the norm codebooks
        - compress: return the results of Algorithm 1 from paper.
    """
    def __init__(self, n_percentile, quantize, true_norm=False, verbose=False):
        # n_percentile -> number of codewords for norm training (default same as Ks)
        self.M = 2
        self.n_percentile, self.true_norm, self.verbose = n_percentile, true_norm, verbose
        self.code_dtype = np.uint8 if n_percentile <= 2 ** 8 \
            else (np.uint16 if n_percentile <= 2 ** 16 else np.uint32)

        self.percentiles = None     # these are the norm codebooks. The direction codebooks exist in the RQ object.
        self.quantize = quantize

    def class_message(self):
        return "NormPQ, percentiles: {}, quantize: {}".format(self.n_percentile, self.quantize.class_message())

    def fit(self, vecs, iter):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert self.n_percentile < N, "the number of norm intervals should be more than Ks"

        norms, normalized_vecs = normalize(vecs)    # get the true norm
        self.quantize.fit(normalized_vecs, iter)    # fit on direction vector (i.e. x' in paper)

        # get the approximation for the relative norm (i.e. l_x in paper)
        compressed_vecs = self.quantize.compress(normalized_vecs)
        temp = np.linalg.norm(compressed_vecs, axis=1)
        norms = norms / np.where(temp < NORM_BUFFER, NORM_BUFFER, temp)
        self.percentiles, _ = kmeans2(norms[:], self.n_percentile, iter=iter, minit='points')
        return self

    def encode_norm(self, norms):
        norm_index, _ = vq(norms[:], self.percentiles)
        return norm_index

    def decode_norm(self, norm_index):
        return self.percentiles[norm_index]

    def compress(self, vecs):
        norms, normalized_vecs = normalize(vecs)
        compressed_vecs = self.quantize.compress(normalized_vecs)   # approximation of direction vectors
        del normalized_vecs

        norms = norms / np.linalg.norm(compressed_vecs, axis=1)
        return (compressed_vecs.transpose() * norms).transpose()


class ResidualPQ(object):
    """
    This class has the same member functions as the PQ class and receives a list of PQs as input. Thus, all functions
    call each PQ object's identically named functions. This means encode/decode have an additional dimension
    for each PQ in the input list.
    """
    def __init__(self, pqs=None, verbose=False):

        assert len(pqs) > 0
        self.verbose = verbose
        self.deep = len(pqs)
        self.code_dtype = pqs[0].code_dtype
        self.M = max([pq.M for pq in pqs])
        self.pqs = pqs

        for pq in self.pqs:
            assert pq.code_dtype == self.code_dtype

    def class_message(self):
        messages = ""
        for i, pq in enumerate(self.pqs):
            messages += pq.class_message()
        return messages

    def fit(self, T, iter, D=None):
        assert T.dtype == np.float32
        assert T.ndim == 2

        vecs = np.empty(shape=T.shape, dtype=T.dtype)
        vecs[:, :] = T[:, :]
        if D is not None:
            vecs_d = np.empty(shape=D.shape, dtype=D.dtype)
            vecs_d[:, :] = D[:, :]

        # training each subspace quantizer
        for layer, pq in enumerate(self.pqs):
            pq.fit(vecs, iter)
            compressed = pq.compress(vecs)
            vecs = vecs - compressed    # fitting on the residuals
            del compressed

            if D is not None:
                compressed_d = pq.compress(vecs_d)
                vecs_d -= compressed_d

            if self.verbose:
                norms = np.linalg.norm(vecs, axis=1)
                print("# layer: {},  residual average norm : {} max norm: {} min norm: {}"
                      .format(layer, np.mean(norms), np.max(norms), np.min(norms)))
        return self

    def encode(self, vecs):
        codes = np.zeros((len(vecs), self.deep, self.M), dtype=self.code_dtype)  # N * deep * M
        for i, pq in enumerate(self.pqs):
            codes[:, i, :pq.M] = pq.encode(vecs)
            vecs = vecs - pq.decode(codes[:, i, :pq.M])
        return codes  # N * deep * M

    def decode(self, codes):
        vecss = [pq.decode(codes[:, i, :pq.M]) for i, pq in enumerate(self.pqs)]
        return np.sum(vecss, axis=0)

    def compress(self, X):
        N, D = np.shape(X)

        sum_residual = np.zeros((N, D), dtype=X.dtype)

        vecs = np.zeros((N, D), dtype=X.dtype)
        vecs[:, :] = X[:, :]

        for i, pq in enumerate(self.pqs):
            compressed = pq.compress(vecs)
            vecs[:, :] = vecs - compressed
            sum_residual[:, :] = sum_residual + compressed
            del compressed
        return sum_residual


class PQ(object):
    """
    This is the base Product Quantizer class responsible for training a single codebook. This subspace is further
    decomposed into M subspaces. The member functions are as follows:
        - fit: finds the Ks centroids that corresponds to the given subspace
        - encode: returns a list of indices that maps each row vector to a centroid
        - decode: modifies original dataset in-place to its corresponding centroid (i.e. codebook-based approximation)
        - compress: encode -> decode
    """
    def __init__(self, M, Ks, verbose=False):
        assert 0 < Ks <= 2 ** 32
        self.M, self.Ks, self.verbose = M, Ks, verbose
        self.code_dtype = np.uint8 if Ks <= 2 ** 8 else (np.uint16 if Ks <= 2 ** 16 else np.uint32)
        self.codewords = None
        self.Ds = None
        self.Dim = -1

    def class_message(self):
        return "Subspace PQ, M: {}, Ks : {}, code_dtype: {}".format(self.M, self.Ks, self.code_dtype)

    def fit(self, vecs, iter):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert self.Ks < N, "the number of training vector should be more than Ks"
        self.Dim = D

        remainder = D % self.M
        quotient = int(D / self.M)
        dims_width = [quotient + 1 if i < remainder else quotient for i in range(self.M)]
        self.Ds = np.cumsum(dims_width)     # prefix sum
        self.Ds = np.insert(self.Ds, 0, 0)  # insert zero at beginning

        # [m][ks][ds]: m-th subspace, ks-the codeword, ds-th dim
        self.codewords = np.zeros((self.M, self.Ks, np.max(dims_width)), dtype=np.float32)
        for m in range(self.M):
            if self.verbose:
                print("#    Training the subspace: {} / {}, {} -> {}".format(m, self.M, self.Ds[m], self.Ds[m+1]))
            vecs_sub = vecs[:, self.Ds[m]:self.Ds[m+1]]
            self.codewords[m, :, :self.Ds[m+1] - self.Ds[m]], _ = kmeans2(vecs_sub, self.Ks, iter=iter, minit='points')

        return self

    def encode(self, vecs):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape

        # codes[n][m] : code of n-th vec, m-th subspace
        codes = np.empty((N, self.M), dtype=self.code_dtype)
        for m in range(self.M):
            vecs_sub = vecs[:, self.Ds[m]: self.Ds[m+1]]
            codes[:, m], _ = vq(vecs_sub, self.codewords[m, :, :self.Ds[m+1] - self.Ds[m]])

        return codes

    def decode(self, codes):
        assert codes.ndim == 2
        N, M = codes.shape
        assert M == self.M
        assert codes.dtype == self.code_dtype

        vecs = np.empty((N, self.Dim), dtype=np.float32)
        for m in range(self.M):
            vecs[:, self.Ds[m]: self.Ds[m+1]] = self.codewords[m, codes[:, m], :self.Ds[m+1] - self.Ds[m]]

        return vecs

    def compress(self, vecs):
        return self.decode(self.encode(vecs))


def chunk_compress(pq, vecs):
    """
    Run pq's .compress function on the vecs in batches to get the codebook approximation. Assumes that pq is fitted.
    By default, the batchsize is the same size as the training set (i.e. compress all at once).
    """
    chunk_size = 1000000    # same size as training set by default
    compressed_vecs = np.empty(shape=vecs.shape, dtype=np.float32)
    for i in range(math.ceil(len(vecs) / chunk_size)):
        compressed_vecs[i * chunk_size: (i + 1) * chunk_size, :] \
            = pq.compress(vecs[i * chunk_size: (i + 1) * chunk_size, :].astype(dtype=np.float32))
    return compressed_vecs


def execute(pq, X, Q, G, top_k=5, train_size=1300):
    """
    Function that preprocesses and queries the dataset.
    Params:
        - pq: the product quantizer
        - X: atoms
        - Q: signals
        - G: groundtruth
    """
    # preprocessing (i.e. creating the codebooks)
    np.random.seed(123)
    pq.fit(X[:train_size].astype(dtype=np.float32), iter=20)
    compressed = chunk_compress(pq, X)
    precisions, budgets = BatchSorter(compressed, Q, G, top_k, batch_size=200).precision()
    return precisions, budgets


if __name__ == '__main__':
    topk = 20
    codebook = 4    # i.e. subspaces
    Ks = 256    # i.e. quantizers per subspace
    num_atoms = 1000
    len_signal = 30

    atoms, signals = get_data(data_type=NETFLIX, num_atoms=num_atoms, len_signal=len_signal)
    assert (atoms.shape[1] == signals.shape[-1]), "atoms and signals are not current shape for matrix multiplication"
    groundtruth = np.argsort(-atoms @ signals.T)
    groundtruth = groundtruth[:topk].T
    assert (groundtruth.shape == (len(signals), topk))

    # X, Q, G is equivalent to atoms, signals, and groundtruth (naive MIPS)
    pqs = [PQ(M=1, Ks=Ks) for _ in range(codebook - 1)]
    quantizer = ResidualPQ(pqs=pqs)
    quantizer = NormPQ(n_percentile=Ks, quantize=quantizer)
    execute(quantizer, atoms, signals.astype('float32'), groundtruth, top_k=5, train_size=num_atoms)