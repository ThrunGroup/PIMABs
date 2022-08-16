import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pandas import read_csv
from utils.constants import (
    COLORS,
    MARKERS,
    GREEDY_MIPS,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    MEDIAN_ELIMINATION,
    LSH_MIPS,
    PCA_MIPS,
    NORMAL_CUSTOM,
    ADVERSARIAL_CUSTOM,
    NORMAL_PAPER,
    ALG_TO_COLOR,
)
from utils.utils import fit_and_plot
from typing import List, Any


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
    data_types: List[str] = [
        NORMAL_CUSTOM,
        ADVERSARIAL_CUSTOM,
        NORMAL_PAPER
    ],
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
        xlabel = "Signal Vector Size"
        ylabel = "Sample Complexity"
        loc = "upper left"
    else:
        path = os.path.join(path, "precision_speed")
        csv_files = glob.glob(os.path.join(path, "*SPEEDUP*"))
        exp = "Precision-Speed Tradeoff Baselines"
        xlabel = "speed (i.e. inverse budget)"
        ylabel = "precision"
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
                if alg_name == ACTION_ELIMINATION or alg_name == ADAPTIVE_ACTION_ELIMINATION:
                    functions = [np.log]
                    function_names = ["Logarithm"]
                elif alg_name == MEDIAN_ELIMINATION:
                    functions = [np.sqrt]
                    function_names = ["Sqrt"]
                elif alg_name == GREEDY_MIPS:
                    functions = [np.array]
                    function_names = ["Linear"]
                else:
                    raise NotImplementedError("invalid alg_name")
                fit_and_plot(x, y, functions, function_names)

            # plot the datapoints
            x_sorted, y_sorted = zip(*sorted(zip(x, y), key=lambda pair: pair[0]))
            if is_scaling:
                plt.scatter(
                    x, y,
                    color=ALG_TO_COLOR[alg_name],
                    label=f"{alg_name}"
                )
                plt.plot(x_sorted, y_sorted, color=ALG_TO_COLOR[alg_name])
            else:
                plt.scatter(
                    x, y,
                    #marker=MARKERS[color_i % num_colors],
                    color=ALG_TO_COLOR[alg_name],
                    label=f"{alg_name}"
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
    create_baseline_plots_from_data(
        is_scaling=True,
        fit=False,
        include_error_bar=True
    )
    # create_baseline_plots_from_data(
    #     #max_speedup=20,
    #     is_scaling=False,
    #     include_error_bar=False,
    #     alg_names=[ADAPTIVE_ACTION_ELIMINATION],
    #     data_types=[NORMAL_PAPER]
    #     #alg_names = [LSH_MIPS, PCA_MIPS]
    # )


