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
    MOVIE_LENS,

    MEDIAN_ELIMINATION,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    GREEDY_MIPS,
    LSH_MIPS,
    PCA_MIPS,
)
from utils.utils import fit_and_plot


def translate_alg_name(alg_name: str):
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
    else:
        raise Exception("Bad algorithm name")


def translate_dataset_name(dataset_name: str):
    if dataset_name == ADVERSARIAL_CUSTOM:
        return "Synthetic Adversarial"
    elif dataset_name == NORMAL_CUSTOM:
        return "Synthetic Gaussian"
    elif dataset_name == NETFLIX:
        return "Netflix"
    elif dataset_name == MOVIE_LENS:
        return "Movie Lens"
    else:
        raise Exception("Bad dataset name {}".format(dataset_name))


def extract_alg(filename: str):
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
    else:
        raise Exception("No algorithm was able to be extracted from " + filname)


def create_baseline_plots_from_data(
    function_list: List[np.ufunc] = None,
    function_name_list: List[str] = None,
    max_speedup: int = None,
    data_types: List[str] = [NORMAL_CUSTOM, ADVERSARIAL_CUSTOM],
    alg_names: List[str] = [],
    is_scaling: bool = True,
    fit: bool = True,
    include_error_bar: bool = False
):
    # get all csv files
    path = os.path.join(os.getcwd(), "logs")
    if is_scaling:
        path = os.path.join(path, "scaling")
        csv_files = glob.glob(os.path.join(path, "*SCALING*"))
        exp = f"Scaling Baselines"
        xlabel = "Signal Vector Size (d)"
        ylabel = "Sample Complexity"
        loc = "upper left"
    else:
        path = os.path.join(path, "precision_speed")
        csv_files = glob.glob(os.path.join(path, "*SPEEDUP*"))
        exp = "Precision-Speed Tradeoff Baselines"
        xlabel = "Speedup"
        ylabel = "Precision at 10"
        loc = "lower right"

    color_i = 0
    num_colors = len(COLORS)

    for datatype in data_types:
        plt.figure(figsize=(6, 4.5))
        title = exp + f": {datatype}"
        color_i = 0
        for f in csv_files:
            alg_name = extract_alg(f)

            not_found = 0
            data = read_csv(f)

            # plot per datatype
            if f.find(datatype) < 0:
                continue

            if alg_name not in alg_names and len(alg_names) > 0:
                continue

            if is_scaling:
                alg_name = f[:f.index("_SCALING")].split("/")[-1]
                x = data["signal_sizes"].tolist()
                y = data["budgets"].tolist()
                error = data["budgets_std"].tolist()
            else:
                alg_name = f[:f.index("_SPEEDUP")].split("/")[-1]
                budget_data = data["budgets"].tolist()

                x = np.max(budget_data) / budget_data  # speed is inverse budget
                y = data["accuracy"].tolist()
                error = data["accuracy_std"].tolist()

            if max_speedup is not None:
                x = np.where(x > max_speedup, max_speedup, x)

            # fit the plots
            if fit and is_scaling:
                if is_logspace:
                    def power10(array):
                        return np.power(10, array)

                    def power10_sqrt(array):
                        return np.power(10, array) ** (1 / 2)

                    functions = [np.array, power10_sqrt, power10]
                    function_names = [
                        "$y \propto log x$",
                        "$y \propto x^{1/2}$",
                        "$y \propto x$",
                    ]
                    x = np.log10(x)
                    xlabel = "signal size"
                else:
                    functions = [np.array, np.sqrt, np.log]
                    function_names = ["Linear fit", "Sqrt fit", "Logarithmic fit"]
                    xlabel = None

                title = f"{translate_alg_name(mips_alg)} on {translate_dataset_name(data_type)}"
                if title_accuracy:
                    title += f", Accuracy: ({accuracy_list.mean():.2f})"

                fit_and_plot(
                    x,
                    y,
                    functions,
                    function_names,
                    ys_std=budgets_std,
                    title=title,
                    xlabel=xlabel,
                    is_logspace=is_logspace,
                )

            # plot the datapoints
            x_sorted, y_sorted = zip(*sorted(zip(x, y), key=lambda pair: pair[0]))
            if is_scaling:
                plt.scatter(
                    x, y,
                    color=ALG_TO_COLOR[alg_name],
                    label=f"{translate_alg_name(alg_name)}"
                )
                plt.plot(x_sorted, y_sorted, color=ALG_TO_COLOR[alg_name])
            else:
                plt.scatter(
                    x, y,
                    #marker=MARKERS[color_i % num_colors],
                    color=ALG_TO_COLOR[alg_name],
                    label=f"{translate_alg_name(alg_name)}"
                )
                plt.plot(x_sorted, y_sorted, color=ALG_TO_COLOR[alg_name])

            # Sort the legend entries (labels and handles) by labels
            handles, labels = plt.gca().get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            plt.legend(handles, labels, loc=loc)

            if include_error_bar:
                plt.errorbar(x, y, error, fmt=".", color="black")

            color_i += 1

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.show()


if __name__ == "__main__":
    # scaling experiments (synthetic and real datasets)
    create_baseline_plots_from_data(
        is_scaling=True,
        data_types=[NORMAL_CUSTOM, ADVERSARIAL_CUSTOM],
        fit=False,
        include_error_bar=True
    )
    create_baseline_plots_from_data(
        is_scaling=True,
        data_types=[NETFLIX, ADVERSARIAL_CUSTOM],
        fit=False,
        include_error_bar=True
    )

    # create_baseline_plots_from_data(
    #     is_scaling=True,
    #     fit=False,
    #     include_error_bar=True
    # )
    # create_baseline_plots_from_data(
    #     #max_speedup=20,
    #     is_scaling=False,
    #     include_error_bar=False,
    #     #alg_names=[ADAPTIVE_ACTION_ELIMINATION],
    #     data_types=[NORMAL_CUSTOM]
    #     #alg_names = [LSH_MIPS, PCA_MIPS]
    # )


