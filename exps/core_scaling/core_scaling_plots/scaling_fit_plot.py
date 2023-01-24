from exps.plot_baselines import create_scaling_plots
from utils.constants import SCALING_BASELINES_DATATYPES, SCALING_BASELINES_ALGORITHMS, ACTION_ELIMINATION, ADAPTIVE_ACTION_ELIMINATION
import os

def scaling_fit_plot(algorithms):
    """
    Function to generate the fitted (linear, sqrt, log) scaling plots. Will output each baseline for each dataset.
    """
    for algorithm in algorithms:
        create_scaling_plots(
            is_fit=True,
            include_error_bar=True,
            data_types=SCALING_BASELINES_DATATYPES,
            alg_names=[algorithm],
            dir_name=os.path.join("core_scaling", "logs"),
            is_logspace=True,
        )


if __name__ == "__main__":
    scaling_fit_plot([ACTION_ELIMINATION])#, ADAPTIVE_ACTION_ELIMINATION])