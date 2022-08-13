import os
import pandas as pd
import numpy as np

from utils.constants import (
    NETFLIX,
    YAHOO,
    INDEX,
    CUST_ID,
    RATING,
    MOVIE_ID,
    NUM_NETFLIX_USERS,
    NUM_NETFLIX_MOVIES,
    NUM_DATA_FILES,
    NUM_YAHOO_USERS,
    NUM_YAHOO_ITEMS,
)

def load_data(filename, dataset=NETFLIX):
    if dataset == YAHOO:
        name_field = ITEM_ID
    elif dataset == NETFLIX:
        name_field= CUST_ID
    else:
        raise ValueError("Invalid dataset")

    return pd.read_csv(
        filename,
        header=None,
        names=[name_field, RATING],
        usecols=[0, 1],
    )


def load_netflix():
    """
    Loads 4 Netflix data .txt files and transforms the data into a numpy ndarray of users by movies,
    with each cell representing a rating

    :return: numpy ndarray of users by movies
    """
    # Join the shards into one
    for idx in range(NUM_DATA_FILES):
        if idx == 0:
            df = load_data(os.path.join("raw_data", "combined_data_" + str(idx + 1) + ".txt"))
        else:
            new_shard = load_data(os.path.join("raw_data", "combined_data_" + str(idx + 1) + ".txt"))
            df = df.append(new_shard)

    df[RATING] = df[RATING].astype(float)
    df.index = np.arange(0, len(df))

    df_nan = pd.DataFrame(pd.isnull(df.Rating))  # TODO: Should this be RATING?
    df_nan = df_nan[df_nan[RATING] == True]
    df_nan = df_nan.reset_index()

    movie_np = []
    movie_id = 1

    # df_nan holds the rows that indicate the start of a movie's ratings
    # i holds values representing the second to the last such rows
    # j holds values representing the first to the second-to-last such rows
    for i, j in zip(df_nan[INDEX][1:], df_nan[INDEX][:-1]):
        # we create a numpy ndarray with the right amount of movie_id elements to match the amount of that movie's ratings
        temp = np.full((1, i - j - 1), movie_id)
        movie_np = np.append(movie_np, temp)
        movie_id += 1

    # Manually handle the last user, which was excluded from the loop above
    # TODO: Merge this into loop above
    last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1), movie_id)
    movie_np = np.append(movie_np, last_record)

    # Keep only non-missing ratings
    df = df[pd.notnull(df[RATING])]
    df[MOVIE_ID] = movie_np.astype(int)
    df[CUST_ID] = df[CUST_ID].astype(int)

    # Assign the appropriate values to X by getting each user's ID and the movie ID
    X = np.ndarray(shape=(NUM_NETFLIX_USERS, NUM_NETFLIX_MOVIES), dtype=float)
    for idx in range(len(df)):
        # Customer IDs and movie IDs are 1-indexed, but numpy arrays are 0-indexed
        cust_id = df[CUST_ID].iloc[idx] - 1
        movie_id = df[MOVIE_ID].iloc[idx] - 1
        rating = df.iloc[idx][RATING]
        X[cust_id][movie_id] = rating

    return X

def load_yahoo():
    """
    Loads Yahoo! Music data .txt file and transforms the data into a numpy ndarray of users by items, with each cell representing a rating

    :return: numpy ndarray of users by items
    """
    raw_fp = os.path.join("raw_data", "trainIdx1.txt")
    cleaned_fp = os.path.join("raw_data", "trainIdx1_modified.txt")
    if not os.path.exists(cleaned_fp):
        with open(raw_fp, "r") as fin:
            data = fin.read()
            data = data.replace("\t", ",")

        with open(cleaned_fp, "w") as fout:
            fout.write(data)

    df = load_data(cleaned_fp)
    df[RATING] = df[RATING].astype(float)
    df.index = np.arange(0, len(df))

    df_nan = pd.DataFrame(pd.isnull(df.Rating))
    df_nan = df_nan[df_nan[RATING] == True]
    # Reset the index to be continguous integers. The old index corresponds to the original ID, and we specifically
    # pass drop=False to keep that value in the new "index" field
    df_nan = df_nan.reset_index(drop=False)

    user_np = []
    user_id = 0

    # df_nan contains rows that have a user rating
    # INDEX corresponds to the original user ID
    # TODO: df_nan has columns <...>
    for i, j in zip(df_nan[INDEX][1:], df_nan[INDEX][:-1]):
        # i holds values representing the second to the last such columns
        # j holds values representing the first to the second-to-last such columns
        # we create a numpy ndarray with the right amount of user_id elements to match the amount of that user's ratings
        # Create a row with each user's ratings
        temp = np.full((1, i - j - 1), user_id)
        user_np = np.append(user_np, temp)
        user_id += 1

    # Manually handle the last user, which was excluded from the loop above
    # TODO: Merge this into loop above
    last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1), user_id)
    user_np = np.append(user_np, last_record)

    # Keep only non-null ratings
    df = df[pd.notnull(df[RATING])]
    df[USER_ID] = user_np.astype(int)
    df[ITEM_ID] = df[ITEM_ID].astype(int)

    # Assign the appropriate values to X by getting each user's ID and the item ID
    X = np.ndarray(shape=(NUM_YAHOO_USERS, NUM_YAHOO_ITEMS), dtype=float)
    for idx in range(len(df)):
        user_id = df[USER_ID].iloc[idx]
        item_id = df[ITEM_ID].iloc[idx]
        rating = df.iloc[idx][RATING]
        X[user_id][item_id] = rating

    return X
