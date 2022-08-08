import pandas as pd
import numpy as np


def load_netflix_data():
    """
    Loads 4 Netflix data .txt files and transforms the data into a numpy ndarray of users by movies, with each cell representing a rating

    :X: numpy ndarray of users by movies
    """
    # please use the simplified files for testing
    df1 = pd.read_csv(
        "combined_data_1.txt",
        header=None,
        names=["Cust_Id", "Rating"],
        usecols=[0, 1],
    )
    df2 = pd.read_csv(
        "combined_data_2.txt",
        header=None,
        names=["Cust_Id", "Rating"],
        usecols=[0, 1],
    )
    df3 = pd.read_csv(
        "combined_data_3.txt",
        header=None,
        names=["Cust_Id", "Rating"],
        usecols=[0, 1],
    )
    df4 = pd.read_csv(
        "combined_data_4.txt",
        header=None,
        names=["Cust_Id", "Rating"],
        usecols=[0, 1],
    )

    df1["Rating"] = df1["Rating"].astype(float)
    df2["Rating"] = df2["Rating"].astype(float)
    df3["Rating"] = df3["Rating"].astype(float)
    df4["Rating"] = df4["Rating"].astype(float)

    df = df1
    df = df.append(df2)
    df = df.append(df3)
    df = df.append(df4)

    df.index = np.arange(0, len(df))

    df_nan = pd.DataFrame(pd.isnull(df.Rating))
    df_nan = df_nan[df_nan["Rating"] == True]
    df_nan = df_nan.reset_index()

    movie_np = []
    movie_id = 1

    for i, j in zip(df_nan["index"][1:], df_nan["index"][:-1]):
        temp = np.full((1, i - j - 1), movie_id)
        movie_np = np.append(movie_np, temp)
        movie_id += 1

    # Account for last record and corresponding length
    last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1), movie_id)
    movie_np = np.append(movie_np, last_record)

    df = df[pd.notnull(df["Rating"])]

    df["Movie_Id"] = movie_np.astype(int)
    df["Cust_Id"] = df["Cust_Id"].astype(int)

    num_users = (
        2649429  # please replace 2649429 with 93986 for testing on the simplified files
    )
    num_movies = 17770
    X = np.ndarray(shape=(num_users, num_movies), dtype=float)
    for idx in range(len(df)):
        cust_id = int(df.iloc[idx]["Cust_Id"]) - 1
        rating = df.iloc[idx]["Rating"]
        movie_id = int(df.iloc[idx]["Movie_Id"]) - 1
        X[cust_id][movie_id] = rating

    return X
