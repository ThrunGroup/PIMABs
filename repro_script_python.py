import os
from exps.run_scaling import run_scaling
from exps.run_tradeoff import run_tradeoff
from exps.plot_baselines import generate_scaling_plots, generate_tradeoff_plots


def main():
    path = os.path.join(os.getcwd(), "logs")
    scaling_path = os.path.join(path, "scaling")
    tradeoff_path = os.path.join(path, "precision_speed")

    # scaling experiments
    # if os.path.exists(scaling_path):
    #     print("=> Scaling log files already exist")
    # else:
    print("=> Creating Scaling log files")
    run_scaling()

    # # precision-speed tradeoff experiments
    # if os.path.exists(tradeoff_path):
    #     print("=> Precision-Speed tradeoff log files already exist")
    # else:
    #     print("=> Creating Precision-Speed log files")
    #     run_tradeoff()

    print("=> Generating scaling plots")
    generate_scaling_plots()

    # print("=> Generating tradeoff plots")
    # generate_tradeoff_plots()


if __name__ == "__main__":
    main()