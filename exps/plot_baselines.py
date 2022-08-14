import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pandas import read_csv
from utils.constants import (
    COLORS,
    GREEDY_MIPS,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
    MEDIAN_ELIMINATION,
)
from utils.utils import fit_and_plot
from typing import List, Any



def create_baseline_plots_from_data(
        function_list: List[np.ufunc] = None,
        function_name_list: List[str] = None,
        is_scaling: bool = True,
        fit: bool = True,
        include_error_bar: bool = False
):
    # get all csv files
    path = os.path.join(os.getcwd(), "logs")
    if is_scaling:
        csv_files = glob.glob(os.path.join(path, "*SCALING*"))
        title = "Scaling Baselines"
        xlabel = "signal vector size"
        ylabel = "budget"
    else:
        csv_files = glob.glob(os.path.join(path, "*SPEEDUP*"))
        title = "Precision-Speed Tradeoff Baselines"
        xlabel = "speed (i.e. inverse budget)"
        ylabel = "precision"

    for i, f in enumerate(csv_files):
        data = read_csv(f)
        if is_scaling:
            alg_name = f[:f.index("_SCALING")].split("/")[-1]
            x = data["signal_sizes"].tolist()
            y = data["budgets"].tolist()
        else:
            alg_name = f[:f.index("_SPEEDUP")].split("/")[-1]
            budget_data = data["budgets"].tolist()
            x = np.max(budget_data) / budget_data  # speeds is inverse budget
            y = data["accuracy"].tolist()

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
        plt.plot(x, y, color=COLORS[i], label=f"{alg_name}")
        plt.legend(loc="upper left")
        if include_error_bar:
            error = data["budgets_std"].tolist()
            plt.errorbar(x, y, error, fmt=".", color="black")

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()


if __name__ == "__main__":
    create_baseline_plots_from_data(is_scaling=True, fit=False, include_error_bar=False)
    #create_baseline_plots_from_data(is_scaling=False, include_error_bar=False)


