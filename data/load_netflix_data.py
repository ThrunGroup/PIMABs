import pandas as pd
import numpy as np
from utils.constants import (
    CUST_ID,
    RATING,
    MOVIE_ID,
    NUM_NETFLIX_USERS,
    NUM_NETFLIX_MOVIES,
    NUM_DATA_FILES,
)


def load_data(filename):
    return pd.read_csv(
        filename,
        header=None,
        names=[CUST_ID, RATING],
        usecols=[0, 1],
    )


def load_netflix_data():
    """
    Loads 4 Netflix data .txt files and transforms the data into a numpy ndarray of users by movies, with each cell representing a rating

    :return: numpy ndarray of users by movies
    """
    # loading four smaller datasets and placing them into the dataframes list
    dataframes = [None for _ in range(NUM_DATA_FILES)]
    for idx in range(NUM_DATA_FILES):
        dataframes[idx] = load_data("combined_data_" + str(idx + 1) + ".txt")

    # converting the rating column of each dataset's dataframe into a float column
    for df_x in dataframes:
        df_x[RATING] = df_x[RATING].astype(float)

    # Combining the four smaller datasets from the four separate files into a full one
    df = dataframes[0]
    for idx in range(1, NUM_DATA_FILES):
        df = df.append(dataframes[idx])

    df.index = np.arange(0, len(df))

    df_nan = pd.DataFrame(pd.isnull(df.Rating))
    df_nan = df_nan[df_nan[RATING] == True]
    # Resetting index to use the old index in below for loop
    df_nan = df_nan.reset_index()

    # helper list to hold column in df for the movie ids
    movie_np = []
    movie_id = 1

    # df_nan holds the rows that indicate the start of a movie's ratings
    # i holds values representing the second to the last such rows
    # j holds values representing the first to the second-to-last such rows
    for i, j in zip(df_nan["index"][1:], df_nan["index"][:-1]):
        # we create a numpy ndarray with the right amount of movie_id elements to match the amount of that movie's ratings
        temp = np.full((1, i - j - 1), movie_id)
        # appending this ndarray to movie_np to help construct column in df for the movie ids
        movie_np = np.append(movie_np, temp)
        # Incrementing movie_id so the correct one is appended to movie_np in the next loop iteration
        movie_id += 1

    # movie_np now has elements for all except the last movie_id
    # so, we create another numpy ndarray with the right amount of the last movie_id elements to match the amount of that movie's ratings
    last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1), movie_id)
    # appending this ndarray to movie_np to finish constructing column in df for the movie ids
    movie_np = np.append(movie_np, last_record)

    # Keep only non-missing ratings
    df = df[pd.notnull(df[RATING])]

    df[MOVIE_ID] = movie_np.astype(int)
    df[CUST_ID] = df[CUST_ID].astype(int)

    X = np.ndarray(shape=(NUM_NETFLIX_USERS, NUM_NETFLIX_MOVIES), dtype=float)
    for idx in range(len(df)):
        # Retrieving customer id of current row and subtracting 1 since customer ids are 1-indexed but numpy ndarrays are 0-indexed
        cust_id = df[CUST_ID].iloc[idx] - 1
        rating = df.iloc[idx][RATING]
        # Retrieving movie id of current row and subtracting 1 since movie ids are 1-indexed but numpy ndarrays are 0-indexed
        movie_id = df[MOVIE_ID].iloc[idx] - 1
        X[cust_id][movie_id] = rating

    return X
