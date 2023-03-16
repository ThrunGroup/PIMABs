import os
import glob

from exps.core_scaling.scaling_baselines import scaling_baselines
from exps.core_scaling.bucket_ae_scaling.bucket_action_elimination_scaling import scaling_bucket_ae, scaling_bucket_ae_plot
from exps.core_scaling.scaling_baselines_plot import scaling_baselines_plot
from exps.core_scaling.scaling_fit_plot import scaling_fit_plot
from exps.core_tradeoff.tradeoff_baselines import tradeoff_baselines, tradeoff_baselines_plot
from exps.plot_baselines import generate_scaling_plots, generate_tradeoff_plots
from exps.large_scaling.large_scaling_exps import large_scaling_exps
from utils.constants import SCALING_BASELINES_ALGORITHMS


def find_absent_algorithms(path, algorithms):
    absent = []
    for algorithm in algorithms:
        algo_logs = glob.glob(os.path.join(path, f"*{algorithm}*"))
        if len(algo_logs) == 0:
            absent.append(algorithm)

    return absent


def main():
    path = os.path.join(os.getcwd(), "exps")
    scaling_path = os.path.join(path, "core_scaling", "logs")
    tradeoff_path = os.path.join(path, "core_tradeoff", "logs")

    # generate scaling/tradeoff log files
    scaling_baselines_algos = find_absent_algorithms(scaling_path, SCALING_BASELINES_ALGORITHMS)

    # Todo @jey, @luke: use this function accordingly... this is just an example.
    tradeoff_baselines_algos = find_absent_algorithms(tradeoff_path, SCALING_BASELINES_ALGORITHMS)

    # Todo @jey, @luke: need to add each of your scaling experiments here!
    # print(f"=> Creating Scaling log files for {scaling_baselines_algos}")
    # scaling_baselines(scaling_baselines_algos)
    scaling_baselines(SCALING_BASELINES_ALGORITHMS)

    # Todo @jey, @luke: need to add each of your tradeoff experiments here (included example below)!
    print(f"=> Creating Precision-Speed log files for {tradeoff_baselines_algos}")
    tradeoff_baselines()

    # generate the relative plots
    # Todo @jey, @luke: add scaling plot code here
    print("=> Generating scaling plots")
    scaling_baselines_plot(SCALING_BASELINES_ALGORITHMS)
    scaling_fit_plot(SCALING_BASELINES_ALGORITHMS)
    scaling_bucket_ae_plot()

    # Todo @jey, @luke: add tradeoff plot code here
    print("=> Generating tradeoff plots")
    tradeoff_baselines_plot()


if __name__ == "__main__":
    main()