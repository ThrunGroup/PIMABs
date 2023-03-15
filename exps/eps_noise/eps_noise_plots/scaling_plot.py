from exps.plot_baselines import create_scaling_plots
from utils.constants import ACTION_ELIMINATION, NETFLIX


def scaling_baselines_plot(algorithms):
    """
    Function to generate the scaling plots of the mips algorithms that were run with scaling_noise.
    The output is one plot with the specified baselines for the Netflix dataset.
    """
    create_scaling_plots(
        is_fit=False,
        include_error_bar=True,
        data_types=[NETFLIX],
        alg_names=[ACTION_ELIMINATION],
        dir_name="eps_noise/logs",
        is_logspace_y=True,
    )


if __name__ == "__main__":
    scaling_baselines_plot([ACTION_ELIMINATION])