from plot_baselines import create_scaling_plots
from utils.constants import (
    NORMAL_CUSTOM,
    NETFLIX,
    ENVIRONMENTAL_SOUND,
    ACTION_ELIMINATION,
    ENVIRONMENTAL_SOUND_DL
)

if __name__ == "__main__":
    create_scaling_plots(
        is_fit=True,
        is_logspace=True,
        include_error_bar=True,
        dir_name="mp_scaling",
        caching_type="",
        data_types=[ENVIRONMENTAL_SOUND_DL],
        alg_names=[ACTION_ELIMINATION]
    )
    # create_scaling_plots(
    #     is_fit=True,
    #     is_logspace=True,
    #     include_error_bar=True,
    #     dir_name="mp_scaling",
    #     caching_type="PI_cache",
    #     data_types=[NETFLIX],
    #     alg_names=[ACTION_ELIMINATION]
    # )