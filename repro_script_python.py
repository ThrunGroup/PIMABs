import os

from exps.scaling.run_scaling import run_scaling
from exps.speedup_accuracy_tradeoff.run_tradeoff import run_tradeoff
from exps.plot_baselines import generate_scaling_plots, generate_tradeoff_plots
from exps.large_scaling.large_scaling_exps import large_scaling_exps


def main():
    exp_path = "exps"

    # scaling experiments
    scaling_path = os.path.join(exp_path, "scaling", "logs")
    if os.path.exists(scaling_path):
        print("=> Scaling log files already exist")
    else:
        print("=> Creating Scaling log files")
        run_scaling()

    # precision-speed speedup_accuracy_tradeoff experiments
    tradeoff_path = os.path.join(exp_path, "speedup_accuracy_tradeoff", "logs")
    if os.path.exists(tradeoff_path):
        print("=> Precision-Speed speedup_accuracy_tradeoff log files already exist")
    else:
        print("=> Creating Precision-Speed log files")
        run_tradeoff()

    print("=> Generating scaling plots")
    generate_scaling_plots()
    print("=> Generating speedup_accuracy_tradeoff plots")
    generate_tradeoff_plots()
    print("=> Doing large scaling experiments")
    # large_scaling_exps()


if __name__ == "__main__":
    main()
    