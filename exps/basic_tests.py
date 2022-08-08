from algorithms.lsh_mips import LshMips
from algorithms.pca_mips import PcaMips
from data.custom_data import generate_normal_data, generate_adversarial_data
import ipdb

def run_pca_mips(data, query, delta=2):
    print("... Running PCA-MIPS")
    PM = PcaMips(data, query, delta)
    PM.preprocess()
    budget, candidate = PM.run()
    print(" =>> Top candidate: ", data[candidate])
    print(" =>> budget: ", budget)

    budget, candidates = PM.run(top_k=3)
    print(" =>> Top 3 candidates: ", data[candidates])
    print(" =>> budget: ", budget)


def run_lsh_mips(type, data, query):
    print(f"... Running LSH-MIPS {type}")
    LM = LshMips(type, data, query)
    budget, candidate = LM.run(type)
    print(" =>> Top candidate: ", data[candidate])
    print(" =>> budget: ", budget)

    budget, candidates = LM.run(type, top_k=3)
    print(" =>> Top 3 candidates: ", data[candidates])
    print(" =>> budget: ", budget)


if __name__=="__main__":
    atoms, signal = generate_normal_data(seed=0)
    run_pca_mips(atoms, signal)
    run_lsh_mips("asymmetric", atoms, signal)
    run_lsh_mips("symmetric", atoms, signal)

