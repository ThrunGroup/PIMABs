NUM_STDS = 1  # A multiplier on the number of standard deviations we use when creating confidence intervals
BATCH_SIZE = 30

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

MARKERS = [
    "o",
    ",",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
]

# MAB Algorithms
MEDIAN_ELIMINATION = "MEDIAN_ELIMINATION"
ACTION_ELIMINATION = "ACTION_ELIMINATION"
ADAPTIVE_ACTION_ELIMINATION = "ADAPTIVE_ACTION_ELIMINATION"
GREEDY_MIPS = "GREEDY_MIPS"
PCA_MIPS = "PCA_MIPS"
LSH_MIPS = "LSH_MIPS"
SYMMETRIC = "SYMMETRIC"
ASYMMETRIC = "ASYMMETRIC"

# Datasets
ADVERSARIAL_CUSTOM = "ADVERSARIAL_CUSTOM"
NORMAL_CUSTOM = "NORMAL_CUSTOM"
UNIFORM_PAPER = "UNIFORM_PAPER"
NORMAL_PAPER = "NORMAL_PAPER"
SYNTHETIC = "SYNTHETIC"

# Concentration Inequality bound
HOEFFDING = "HOEFFDING"

# Scaling Experiments
NUM_SEEDS = 5
SCALING_NUM_EXPERIMENTS = 10  # corresponds to the number of points on the plot
SCALING_NUM_ATOMS = 1000
SCALING_NUM_SIGNALS = 3  # this is to get the confidence intervals
SCALING_SIZE_MINMAX = (
    100, 10 ** 4,
)
SCALING_EPSILON = 0.0
SCALING_DELTA = 0.3
SCALING_DEPTH = 2
SCALING_TOPK = 10
SCALING_NUM_HFUNC = 3
SCALING_NUM_TABLES = 5

# Precision-Speed Tradeoff Experiments
TRADEOFF_NUM_EXPERIMENTS = 10  # corresponds to the number of points on the plot
TRADEOFF_NUM_ATOMS = 1000
TRADEOFF_DIMENSION = 10000
TRADEOFF_NUM_SIGNALS = 3
TRADEOFF_DELTA = 0.3
TRADEOFF_TOPK = 10
GREEDY_BUDGET_MINMAX = (10, 50)
TRADEOFF_VAR = 5

TRADEOFF_PCA_NUM_EXPERIMENTS = 4  # this is consistent with the paper
TRADEOFF_DEPTH_MINMAX = (0, 4)

TRADEOFF_LSH_NUM_EXPERIMENTS = 24  # this is consistent with the paper
TRADEOFF_HFUNC_MINMAX = (0, 10)
TRADEOFF_TABLES_MINMAX = (1, 4)
