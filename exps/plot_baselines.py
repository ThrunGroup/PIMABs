import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pandas import read_csv
from typing import List, Any
from utils.constants import (
    COLORS,
    ALG_TO_COLOR,
    NORMAL_CUSTOM,
    ADVERSARIAL_CUSTOM,
    NETFLIX,
    NETFLIX_TRANSPOSE,
    MOVIE_LENS,
    COR_NORMAL_CUSTOM,
    CRYPTO_PAIRS,
    CLEAR_LEADER_HARD,
    CLEAR_LEADER_SOFT,
    NO_CLEAR_LEADER,
    MNIST_T,
    SIMPLE_SONG,
    SIFT_1M,

    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    GREEDY_MIPS,
    LSH_MIPS,
    PCA_MIPS,
    NEQ_MIPS,

    MAX_SPEEDUP,
    TRADEOFF_TOPK,
    MAX_SPEEDUP,
    BUCKET_ACTION_ELIMINATION,
    DIMENSION_OF_ATOMS,
    NUMBER_OF_ATOMS,
)
from utils.utils import fit_and_plot


def translate_alg_name(alg_name: str):
    """
    Translates algorithm names from the codebase into algorithm names consistent with the paper.

    :param alg_name: the algorithm name used throughout the code
    :returns: the translated algorithm name.
    """
    if alg_name == ACTION_ELIMINATION:
        return "BanditMIPS"
    elif alg_name == ADAPTIVE_ACTION_ELIMINATION:
        return "BanditMIPS-$\\alpha$"
    elif alg_name == MEDIAN_ELIMINATION:
        return "BoundedME"
    elif alg_name == GREEDY_MIPS:
        return "GREEDY-MIPS"
    elif alg_name == LSH_MIPS:
        return "LSH-MIPS"
    elif alg_name == PCA_MIPS:
        return "PCA-MIPS"
    elif alg_name == NEQ_MIPS:
        return "NEQ_MIPS"
    elif alg_name == BUCKET_ACTION_ELIMINATION:
        return "Bucket_AE"
    else:
        raise Exception("Bad algorithm name")


def translate_dataset_name(dataset_name: str):
    """
    Translates the dataset names from the codebase into dataset names consistent with the paper.

    :param dataset_name: the dataset name used throughout the code
    :returns: the translated dataset name.
    """
    if dataset_name == ADVERSARIAL_CUSTOM:
        # return "Synthetic Adversarial"
        return ADVERSARIAL_CUSTOM
    elif dataset_name == NORMAL_CUSTOM:
        # return "Synthetic Gaussian"
        return NORMAL_CUSTOM
    elif dataset_name == NETFLIX:
        return "Netflix"
    elif dataset_name == NETFLIX_TRANSPOSE:
        return "Netflix_transposed"
    elif dataset_name == MOVIE_LENS:
        return "Movie Lens"
    elif dataset_name == COR_NORMAL_CUSTOM:
        return COR_NORMAL_CUSTOM
    elif dataset_name == CRYPTO_PAIRS:
        return "Crypto Pairs"
    elif dataset_name == CLEAR_LEADER_HARD:
        return "Clear Leader (Hard)"
    elif dataset_name == CLEAR_LEADER_SOFT:
        return "Clear Leader (Soft)"
    elif dataset_name == NO_CLEAR_LEADER:
        return "No Clear Leader"
    elif dataset_name == MNIST_T:
        return "MNIST Transposed"
    elif dataset_name == SIMPLE_SONG:
        return "Simple Song"
    elif dataset_name == SIFT_1M:
        return "SIFT 1M"
    else:
        raise Exception("Bad dataset name {}".format(dataset_name))


def extract_alg(filename: str):
    """
    Extracts algorithm name from given filename. Used to search for specific algorithms in the log files.
    """
    if ADAPTIVE_ACTION_ELIMINATION in filename:
        return ADAPTIVE_ACTION_ELIMINATION
    elif ACTION_ELIMINATION in filename:
        return ACTION_ELIMINATION
    elif MEDIAN_ELIMINATION in filename:
        return MEDIAN_ELIMINATION
    elif GREEDY_MIPS in filename:
        return GREEDY_MIPS
    elif LSH_MIPS in filename:
        return LSH_MIPS
    elif PCA_MIPS in filename:
        return PCA_MIPS
    elif NEQ_MIPS in filename:
        return NEQ_MIPS
    elif BUCKET_ACTION_ELIMINATION in filename:
        return BUCKET_ACTION_ELIMINATION
    else:
        raise Exception("No algorithm was able to be extracted from " + filename)


def power10(array):
    return np.power(10, array)


def power10_sqrt(array):
    return np.power(10, array) ** (1 / 2)


def create_scaling_plots(
    data_types: List[str] = [],
    alg_names: List[str] = [],
    ind_variables: List[str] = [DIMENSION_OF_ATOMS],
    include_error_bar: bool = False,
    is_fit: bool = False,
    is_logspace: bool = False,
    is_plot_accuracy: bool = False,
    dirname: str = None,
):
    """
    Plot the scaling experiments from the data stored in the logs file.

    :param data_types: the datasets that you want to plot. If empty, warning is triggered.
    :param alg_names: the algorithms that you want to plot. If empty, warning is triggered.
    :param ind_variables: Independent variables (N or D)
    :param include_error_bar: shows the standard deviation
    :param is_fit: whether to fit the plots or not
    :param is_logspace: whether to plot in logspace or not
    :param is_plot_accuracy: whether to annotate accuracy in the graph
    :param dirname: directory name of log files
    """
    # get all csv files
    if dirname is None:
        current_file_directory = os.path.dirname(__file__)
        path = (
            os.path.join(current_file_directory, "logs", "scaling")
            if dirname is None
            else dirname
        )
    else:
        path = dirname

    if len(data_types) == 0:
        raise Exception("At least one dataset must be specified")

    if len(alg_names) == 0:
        raise Exception("At least one algorithm must be specified")

    for ind_variable in ind_variables:
        for datatype in data_types:
            csv_files = glob.glob(
                os.path.join(path, f"*SCALING_{datatype}*ind_var{ind_variable}*")
            )
            plt.figure(figsize=(6, 4.5))
            title = f"Scaling Baselines: {translate_dataset_name(datatype)}"
            for f in csv_files:
                alg_name = extract_alg(f)
                data = read_csv(f)
                independent_var = (
                    "signal_sizes" if "signal_sizes" in data.keys() else "num_atoms"
                )

                # only get relevant data
                if f.find(datatype) < 0 or alg_name not in alg_names:
                    continue

                x = data[independent_var].tolist()
                y = data["budgets"].tolist()
                error = data["budgets_std"].tolist()
                x, y = zip(*sorted(zip(x, y), key=lambda pair: pair[0]))  # sort
                if is_fit:
                    title = f"{translate_alg_name(alg_name)} on {translate_dataset_name(datatype)}"
                    if is_logspace:
                        functions = [np.array, power10_sqrt, power10]
                        function_names = [
                            "$y \propto log x$",
                            "$y \propto x^{1/2}$",
                            "$y \propto x$",
                        ]
                        x = np.log10(x)
                    else:
                        functions = [np.array, np.sqrt, np.log]
                        function_names = ["Linear fit", "Sqrt fit", "Logarithmic fit"]

                    fit_and_plot(
                        xs=x,
                        ys=y,
                        function_list=functions,
                        function_name_list=function_names,
                        ys_std=error,
                        is_logspace=is_logspace,
                    )
                    plt.title(title)
                    xlabel = (
                        "Signal Vector Size (d)"
                        if independent_var == "signal_sizes"
                        else "Number of atoms (N)"
                    )
                    ylabel = "Sample Complexity"
                    plt.ylabel(ylabel)
                    plt.xlabel(xlabel)

                else:
                    plt.scatter(
                        x,
                        y,
                        color=ALG_TO_COLOR[alg_name],
                        label=f"{translate_alg_name(alg_name)}",
                    )
                    plt.plot(x, y, color=ALG_TO_COLOR[alg_name])

                    # Sort the legend entries (labels and handles) by labels
                    handles, labels = plt.gca().get_legend_handles_labels()
                    labels, handles = zip(
                        *sorted(zip(labels, handles), key=lambda t: t[0])
                    )
                    plt.legend(handles, labels, loc="upper left")

                    if include_error_bar:
                        plt.errorbar(x, y, error, fmt=".", color="black")
                    plt.title(title)
                    xlabel = (
                        "Signal Vector Size (d)"
                        if independent_var == "signal_sizes"
                        else "Number of atoms (N)"
                    )
                    ylabel = "Sample Complexity"
                    plt.ylabel(ylabel)
                    plt.xlabel(xlabel)

                if is_plot_accuracy:
                    avg_accuracy = np.mean(data["accuracy"])
                    plt.text(
                        x[-1],
                        y[-1],
                        f"Accuracy: {avg_accuracy}",
                        ha="right",
                        va="bottom",
                    )

            plt.show()


def create_tradeoff_plots(
    data_types: List[str] = [],
    alg_names: List[str] = [],
    top_k: int = 1,
    include_error_bar: bool = False,
    max_speedup: int = None,
):
    """
    Plot the tradeoff experiments from the data stored in the logs file.

    :data_types: the datasets that you want to plot. If empty, warning is triggered.
    :alg_names: the algorithms that you want to plot. If empty, warning is triggered.
    :include_error_bar: shows the standard deviation
    :max_speedup: only displays speedup up to max_speedup
    """
    # get all csv files
    current_file_directory = os.path.dirname(__file__)
    path = os.path.join(current_file_directory, "logs")
    path = os.path.join(path, "precision_speed")
    csv_files = glob.glob(os.path.join(path, "*SPEEDUP*"))

    if len(data_types) == 0:
        raise Exception("At least one dataset must be specified")
    if len(alg_names) == 0:
        raise Exception("At least one algorithm must be specified")

    for datatype in data_types:
        plt.figure(figsize=(6, 4.5))
        title = (
            f"Precision-Speed Tradeoff Baselines: {translate_dataset_name(datatype)}"
        )
        for f in csv_files:
            alg_name = extract_alg(f)
            data = read_csv(f)

            # only get relevant data
            if (
                f.find(datatype) < 0
                or alg_name not in alg_names
                or f.find(str("topk") + str(top_k)) < 0
            ):
                continue

            budget_data = data["budgets"].tolist()
            x = data["naive_budget"] / budget_data  # speed is inverse budget
            y = data["accuracy"].tolist()
            error = data["accuracy_std"].tolist()
            if max_speedup is not None:
                x = np.where(x > max_speedup, max_speedup, x)
            x, y = zip(*sorted(zip(x, y), key=lambda pair: pair[0]))

            # plot the datapoints
            plt.scatter(
                x,
                y,
                color=ALG_TO_COLOR[alg_name],
                label=f"{translate_alg_name(alg_name)}",
            )
            plt.plot(x, y, color=ALG_TO_COLOR[alg_name])

            # Sort the legend entries (labels and handles) by labels
            handles, labels = plt.gca().get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            plt.legend(handles, labels, loc="lower right")

            if include_error_bar:
                plt.errorbar(x, y, error, fmt=".", color="black")

            plt.title(title)
            xlabel = "Speedup"
            ylabel = f"Precision at {top_k}"
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

        fig = plt.gcf()
        fig.set_size_inches(6, 4.5)
        plt.show()


def generate_scaling_plots():
    """
    Generate scaling plots for the two synthetic datasets (normal custom, adversarial custom) and
    the two real-world datsets (netflix and movie lens). This function is called by repro_script_python.py.
    """
    # basic scaling plots (synthetic and real datasets)
    create_scaling_plots(
        is_fit=False,
        include_error_bar=True,
        data_types=[
            NORMAL_CUSTOM,
            ADVERSARIAL_CUSTOM,
            NETFLIX,
            MOVIE_LENS,
            COR_NORMAL_CUSTOM,
        ],
        alg_names=[
            ADAPTIVE_ACTION_ELIMINATION,
            ACTION_ELIMINATION,
            MEDIAN_ELIMINATION,
            GREEDY_MIPS,
            PCA_MIPS,
            LSH_MIPS,
            NEQ_MIPS,
            BUCKET_ACTION_ELIMINATION,
        ],
    )

    # fit the bandit algorithms and greedy-mips for better comparison
    create_scaling_plots(
        is_fit=True,
        is_logspace=True,
        include_error_bar=True,
        data_types=[NETFLIX, MOVIE_LENS],
        alg_names=[
            ADAPTIVE_ACTION_ELIMINATION,
            ACTION_ELIMINATION,
            MEDIAN_ELIMINATION,
            GREEDY_MIPS,
        ],
    )


def generate_tradeoff_plots(top_k: int = 1):
    """
    Generate tradeoff plots for the two synthetic datasets (normal custom, adversarial custom) and
    the two real-world datsets (netflix and movie lens). This function is called by repro_script_python.py.
    """
    # precision-accuracy tradeoff experiments (synthetic and real datasets)
    create_tradeoff_plots(
        data_types=[NORMAL_CUSTOM, NETFLIX, MOVIE_LENS],
        alg_names=[
            ADAPTIVE_ACTION_ELIMINATION,
            ACTION_ELIMINATION,
            MEDIAN_ELIMINATION,
            GREEDY_MIPS,
            PCA_MIPS,
            LSH_MIPS,
            NEQ_MIPS,
        ],
        top_k=top_k,
        include_error_bar=False,
    )
    create_tradeoff_plots(
        data_types=[ADVERSARIAL_CUSTOM],
        alg_names=[
            ADAPTIVE_ACTION_ELIMINATION,
            ACTION_ELIMINATION,
            MEDIAN_ELIMINATION,
            GREEDY_MIPS,
            PCA_MIPS,
            LSH_MIPS,
            NEQ_MIPS,
        ],
        top_k=top_k,
        max_speedup=MAX_SPEEDUP,
        include_error_bar=False,
    )


if __name__ == "__main__":
    generate_scaling_plots()
    generate_tradeoff_plots(top_k=5)
    generate_tradeoff_plots(top_k=20)
