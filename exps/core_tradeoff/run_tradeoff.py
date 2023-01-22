import os

from exps.speedup_precision_exps import speedup_precision_exps
from exps.plot_baselines import create_tradeoff_plots
from neq_mips.constants import TRADEOFF_NEQ_NUM_EXPERIMENTS
from utils.constants import (
    # datasets
    NORMAL_CUSTOM,
    NETFLIX,
    MOVIE_LENS,
    COR_NORMAL_CUSTOM,
    # algorithms
    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    H2ALSH,
    GREEDY_MIPS,
    LSH_MIPS,
    PCA_MIPS,
    NEQ_MIPS,
    ASYMMETRIC,
    HNSW_MIPS,
    NAPG_MIPS,
    SYMMETRIC,
    # NEQ_MIPS,
    # tradeoff specific constants
    TRADEOFF_NUM_EXPERIMENTS,
    TRADEOFF_NUM_ATOMS,
    TRADEOFF_NUM_SIGNALS,
    TRADEOFF_PCA_NUM_EXPERIMENTS,
    TRADEOFF_LSH_NUM_EXPERIMENTS,
    TRADEOFF_DIMENSION,
    NUM_ATOMS_REAL,
    LEN_SIGNAL_NETFLIX,
    LEN_SIGNAL_MOVIE,
)


def run_tradeoff(run: bool = True, plot: bool = True):
    """
    Run speedup_precision tradeoff experiment for the two synthetic datasets (normal custom, adversarial custom) and
    the two real-world datasets (netflix and movie lens). This function is called by repro_script_python.py.

    :param run: Whether to run the experiments
    :param plot: Whether to plot the results
    """
    algorithms = [
        ACTION_ELIMINATION,
        ADAPTIVE_ACTION_ELIMINATION,
        MEDIAN_ELIMINATION,
        GREEDY_MIPS,
        LSH_MIPS,
        H2ALSH,
        NEQ_MIPS,
        PCA_MIPS,
        NAPG_MIPS,
        HNSW_MIPS,
    ]
    data_types = [MOVIE_LENS, NETFLIX, NORMAL_CUSTOM, COR_NORMAL_CUSTOM]
    num_seeds = 1

    # Get data for the scaling experiments for the datasets (three synthetic datasets)
    for top_k in (1, 5, 10):
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(parent_dir, "normalized_logs", f"topk_{top_k}")

        if run:
            for data_type in data_types:  # NORMAL_CUSTOM,
                if data_type is NETFLIX:
                    len_signals = LEN_SIGNAL_NETFLIX
                elif data_type is MOVIE_LENS:
                    len_signals = LEN_SIGNAL_MOVIE
                else:
                    len_signals = TRADEOFF_DIMENSION

                for alg in algorithms:
                    speedup_precision_exps(
                        num_atoms=TRADEOFF_NUM_ATOMS,
                        num_experiments=TRADEOFF_NUM_EXPERIMENTS,
                        len_signals=len_signals,
                        num_signals=30,
                        data_type=data_type,
                        mips_alg=alg,
                        with_replacement=False,
                        dir_name=log_dir,
                        num_best_atoms=top_k,
                        num_seeds=num_seeds,
                    )

        if plot:
            create_tradeoff_plots(
                alg_names=algorithms,
                data_types=data_types,
                top_k=top_k,
                log_dir=log_dir,
                max_speedup=None,
                is_logspace=True,
            )

    # Large dims tradeoff plot for NEQ, BANDIT_MIPS algorithms
    algorithms = [
        ACTION_ELIMINATION,
        ADAPTIVE_ACTION_ELIMINATION,
        MEDIAN_ELIMINATION,
        NEQ_MIPS,
    ]
    data_types = [NETFLIX, NORMAL_CUSTOM, COR_NORMAL_CUSTOM]
    len_signals = 10 ** 5

    for top_k in (1, 5, 10):
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(parent_dir, "large_dim_logs", f"topk_{top_k}")

        if run:
            for data_type in data_types:
                for alg in algorithms:
                    speedup_precision_exps(
                        num_atoms=TRADEOFF_NUM_ATOMS,
                        num_experiments=TRADEOFF_NUM_EXPERIMENTS,
                        len_signals=len_signals,
                        num_signals=30,
                        data_type=data_type,
                        mips_alg=alg,
                        with_replacement=False,
                        dir_name=log_dir,
                        num_best_atoms=top_k,
                        num_seeds=num_seeds,
                    )

        if plot:
            create_tradeoff_plots(
                alg_names=algorithms,
                data_types=data_types,
                top_k=top_k,
                log_dir=log_dir,
                max_speedup=None,
                is_logspace=True,
            )


if __name__ == "__main__":
    run_tradeoff(run=False)
