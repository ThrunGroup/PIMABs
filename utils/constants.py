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

# MAB Algorithms
MEDIAN_ELIMINATION = "MEDIAN_ELIMINATION"
ACTION_ELIMINATION = "ACTION_ELIMINATION"

# Datasets
ADVERSARIAL_CUSTOM = "ADVERSARIAL_CUSTOM"
NORMAL_CUSTOM = "NORMAL_CUSTOM"

# Concentration Inequality bound
HOEFFDING = "HOEFFDING"

# DataFrame columns for Netflix and Yahoo! Music datasets
CUST_ID = "Cust_id"
RATING = "Rating"
MOVIE_ID = "Movie_Id"
USER_ID = "User_id"
ITEM_ID = "Item_id"

# Netflix dataset-specific constants
NUM_NETFLIX_USERS = 2649429
NUM_NETFLIX_MOVIES = 17770
NUM_DATA_FILES = 4

# Yahoo! Music dataset-specific constants
NUM_YAHOO_USERS = 1000990
NUM_YAHOO_ITEMS = 624961
