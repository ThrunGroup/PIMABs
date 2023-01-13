from algorithms.quantization import ResidualPQ, NormPQ, execute


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
    execute(quantizer, atoms, signals.astype('float32'), groundtruth, train_size=num_atoms)