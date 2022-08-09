import pandas as pd
import numpy as np
from constants import CUST_ID, RATING, MOVIE_ID


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

    :X: numpy ndarray of users by movies
    """
    # please use the simplified files for testing
    df1 = load_data("combined_data_1.txt")
    df2 = load_data("combined_data_2.txt")
    df3 = load_data("combined_data_3.txt")
    df4 = load_data("combined_data_4.txt")

    df1[RATING] = df1[RATING].astype(float)
    df2[RATING] = df2[RATING].astype(float)
    df3[RATING] = df3[RATING].astype(float)
    df4[RATING] = df4[RATING].astype(float)

    # Combining the four smaller datasets from the four separate files into a full one
    df = df1
    df = df.append(df2)
    df = df.append(df3)
    df = df.append(df4)

    df.index = np.arange(0, len(df))

    df_nan = pd.DataFrame(pd.isnull(df.Rating))
    df_nan = df_nan[df_nan[RATING] == True]
    # Resetting index to use the old index in below for loop
    df_nan = df_nan.reset_index()

    movie_np = []
    movie_id = 1

    for i, j in zip(df_nan["index"][1:], df_nan["index"][:-1]):
        temp = np.full((1, i - j - 1), movie_id)
        movie_np = np.append(movie_np, temp)
        # Incrementing movie_id so the correct one is appended to movie_np in the next loop iteration
        movie_id += 1

    # Account for last record and corresponding length
    last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1), movie_id)
    movie_np = np.append(movie_np, last_record)

    # Keep only non-missing ratings
    df = df[pd.notnull(df[RATING])]

    df[MOVIE_ID] = movie_np.astype(int)
    df[CUST_ID] = df[CUST_ID].astype(int)

    num_users = (
        2649429  # please replace 2649429 with 93986 for testing on the simplified files
    )
    num_movies = 17770
    X = np.ndarray(shape=(num_users, num_movies), dtype=float)
    for idx in range(len(df)):
        # Retrieving customer id of current row and subtracting 1 since customer ids are 1-indexed but numpy ndarrays are 0-indexed
        cust_id = df[CUST_ID].iloc[idx] - 1
        rating = df.iloc[idx][RATING]
        # Retrieving movie id of current row and subtracting 1 since movie ids are 1-indexed but numpy ndarrays are 0-indexed
        movie_id = df[MOVIE_ID].iloc[idx] - 1
        X[cust_id][movie_id] = rating

    return X
