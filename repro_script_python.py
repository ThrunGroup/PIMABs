import os

from exps.core_scaling.scaling_baselines import scaling_baselines
from exps.core_tradeoff.tradeoff_baselines import tradeoff_baselines
from exps.plot_baselines import generate_scaling_plots, generate_tradeoff_plots
from exps.large_scaling.large_scaling_exps import large_scaling_exps


def main():
    path = os.path.join(os.getcwd(), "exps")
    scaling_path = os.path.join(path, "core_scaling/logs")
    tradeoff_path = os.path.join(path, "core_tradeoff/logs")

    # scaling experiments
    if os.path.exists(scaling_path):
        print("=> Scaling log files already exist")
    else:
        # Todo @jey, @luke: need to add each of your scaling experiments here!
        print("=> Creating Scaling log files")
        scaling_baselines()

    # precision-speed tradeoff experiments
    if os.path.exists(tradeoff_path):
        print("=> Precision-Speed tradeoff log files already exist")
    else:
        # Todo @jey, @luke: need to add each of your tradeoff experiments here!
        print("=> Creating Precision-Speed log files")
        tradeoff_baselines()

    print("=> Generating scaling plots")
    generate_scaling_plots()
    print("=> Generating tradeoff plots")
    generate_tradeoff_plots()


if __name__ == "__main__":
    main()
    