from plot_baselines import create_scaling_plots
from utils.constants import (
    NETFLIX,
    ACTION_ELIMINATION,
)

if __name__ == "__main__":
    create_scaling_plots(
        is_fit=True,
        is_logspace=True,
        include_error_bar=True,
        dir_name="mp_scaling",
        caching_type="(PI cache)",
        data_types=[NETFLIX],
        alg_names=[ACTION_ELIMINATION]
    )