import os

from exps.plot_baselines import create_scaling_plots
from utils.constants import (
    SCALING_BASELINES_DATATYPES,
    SCALING_BASELINES_ALGORITHMS,
    ACTION_ELIMINATION,
    ADAPTIVE_ACTION_ELIMINATION,
)


def scaling_fit_plot(
        algorithms,
        datatypes=SCALING_BASELINES_DATATYPES,
        dir_name=None,
):
    """
    Function to generate the fitted (linear, sqrt, log) scaling plots. Will output each baseline for each dataset.
    """
    if dir_name is None:
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        dir_name = os.path.join(parent_dir, "logs")
    print(dir_name)

    for algorithm in algorithms:
        create_scaling_plots(
            is_fit=True,
            include_error_bar=True,
            data_types=datatypes,
            alg_names=[algorithm],
            dir_name=dir_name,
        )


if __name__ == "__main__":
    scaling_fit_plot([ACTION_ELIMINATION])#, ADAPTIVE_ACTION_ELIMINATION])