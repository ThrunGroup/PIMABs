import pandas as pd
import numpy as np
from utils.constants import (
    USER_ID,
    ITEM_ID,
    RATING,
    NUM_YAHOO_USERS,
    NUM_YAHOO_ITEMS,
)


def load_data(filename):
    return pd.read_csv(
        filename,
        header=None,
        names=[ITEM_ID, RATING],
        usecols=[0, 1],
    )


def load_yahoo_music_data():
    """
    Loads Yahoo! Music data .txt file and transforms the data into a numpy ndarray of users by items, with each cell representing a rating

    :return: numpy ndarray of users by items
    """
    # Opening Yahoo! Music data .txt file in read only mode
    with open(r"trainIdx1.txt", "r") as file:
        data = file.read()
        # Replacing all tabs with commas in the content of file
        data = data.replace("\t", ",")

    # Opening new modified Yahoo! Music data .txt file in write only mode
    with open(r"trainIdx1_modified.txt", "w") as file:
        # Writing the replaced data in the file
        file.write(data)

    # loading the modified dataset and assigning df to it
    df = load_data("trainIdx1_modified.txt")

    df[RATING] = df[RATING].astype(float)

    df.index = np.arange(0, len(df))

    df_nan = pd.DataFrame(pd.isnull(df.Rating))
    df_nan = df_nan[df_nan[RATING] == True]
    # Resetting index to use the old index in below for loop
    df_nan = df_nan.reset_index()

    # helper list to hold column in df for the user ids
    user_np = []
    user_id = 0

    # df_nan holds the rows that indicate the start of a user's ratings
    # i holds values representing the second to the last such rows
    # j holds values representing the first to the second-to-last such rows
    for i, j in zip(df_nan["index"][1:], df_nan["index"][:-1]):
        # we create a numpy ndarray with the right amount of user_id elements to match the amount of that user's ratings
        temp = np.full((1, i - j - 1), user_id)
        # appending this ndarray to user_np to help construct column in df for the user ids
        user_np = np.append(user_np, temp)
        # Incrementing user_id so the correct one is appended to user_np in the next loop iteration
        user_id += 1

    # user_np now has elements for all except the last user_id
    # so, we create another numpy ndarray with the right amount of the last user_id elements to match the amount of that user's ratings
    last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1), user_id)
    # appending this ndarray to user_np to finish constructing column in df for the user ids
    user_np = np.append(user_np, last_record)

    # Keep only non-missing ratings
    df = df[pd.notnull(df[RATING])]

    df[USER_ID] = user_np.astype(int)
    df[ITEM_ID] = df[ITEM_ID].astype(int)

    X = np.ndarray(shape=(NUM_YAHOO_USERS, NUM_YAHOO_ITEMS), dtype=float)
    for idx in range(len(df)):
        # Retrieving user id of current row
        user_id = df[USER_ID].iloc[idx]
        rating = df.iloc[idx][RATING]
        # Retrieving item id of current row
        item_id = df[ITEM_ID].iloc[idx]
        X[user_id][item_id] = rating

    return X
