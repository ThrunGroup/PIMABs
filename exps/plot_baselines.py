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
)
from utils.utils import fit_and_plot
from typing import List, Any


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
        xlabel = "signal vector size"
        ylabel = "budget"
        loc = "upper left"
    else:
        path = os.path.join(path, "precision_speed")
        csv_files = glob.glob(os.path.join(path, "*SPEEDUP*"))
        exp = "Precision-Speed Tradeoff Baselines"
        xlabel = "speed (i.e. inverse budget)"
        ylabel = "precision"
        loc = "upper right"

    color_i = 0
    num_colors = len(COLORS)
    for datatype in data_types:
        title = exp + f": {datatype}"
        color_i = 0
        for f in csv_files:
            not_found = 0
            data = read_csv(f)

            # plot per datatype
            if f.find(datatype) < 0:
                continue

            for alg_name in alg_names:
                if f.find(alg_name) < 0:
                    not_found += 1
            if not_found == len(alg_names) and not_found > 0:
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
            if is_scaling:
                plt.scatter(
                    x, y,
                    color=COLORS[color_i % num_colors],
                    label=f"{alg_name}"
                )
            else:
                plt.scatter(
                    x, y,
                    #marker=MARKERS[color_i % num_colors],
                    color=COLORS[color_i % num_colors],
                    label=f"{alg_name}"
                )
            plt.legend(loc=loc)
            if include_error_bar:
                plt.errorbar(x, y, error, fmt=".", color="black")
            color_i += 1

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.show()


if __name__ == "__main__":
    # create_baseline_plots_from_data(
    #     is_scaling=True,
    #     fit=False,
    #     include_error_bar=True
    # )
    create_baseline_plots_from_data(
        max_speedup=20,
        is_scaling=False,
        include_error_bar=False,
        alg_names=[ADAPTIVE_ACTION_ELIMINATION, ACTION_ELIMINATION, GREEDY_MIPS, MEDIAN_ELIMINATION]
        #alg_names = [LSH_MIPS, PCA_MIPS]
    )


