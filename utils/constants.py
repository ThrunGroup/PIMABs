NUM_STDS = 1  # A multiplier on the number of standard deviations we use when creating confidence intervals
BATCH_SIZE = 30

COLORS = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
]

# MAB Algorithms
MEDIAN_ELIMINATION = "MEDIAN_ELIMINATION"
ACTION_ELIMINATION = "ACTION_ELIMINATION"

# Datasets
ADVERSARIAL_CUSTOM = "ADVERSARIAL_CUSTOM"
NORMAL_CUSTOM = "NORMAL_CUSTOM"
NORMAL_PAPER = "NORMAL_PAPER"

# Concentration Inequality bound
HOEFFDING = "HOEFFDING"

# Baseline
BUFFER = 0.01
PCA = "PCA"
LSH = "LSH"
ASYMMETRIC = "ASYMMETRIC"
SYMMETRIC = "SYMMETRIC"

NUM_POINTS_PCA = 5
NUM_POINTS_LSH = 24
NUM_POINTS_SCALING = 10
NUM_HFUNC = 10
NUM_EXPERIMENTS = 10
