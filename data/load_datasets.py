import os
import numpy as np
from utils.constants import (
    NUM_NETFLIX_USERS,
    NUM_NETFLIX_MOVIES,
    NUM_DATA_FILES,
    NUM_YAHOO_USERS,
    NUM_YAHOO_ITEMS,
    SAMPLE_SIZE_FOR_N,
    SEED,
)


def initialize_sampling_necessities(
    upper_limit: int = NUM_NETFLIX_MOVIES, num_users: int = NUM_NETFLIX_USERS
):
    """
    Initializes numpy ndarray of SAMPLE_SIZE_FOR_N randomly chosen numbers in the appropriate range and empty ndarray to hold the users' ratings of movies or items
    :param upper_limit: Upper limit of random numbers (not inclusive)
    :param num_users: Number of users
    :return: ndarray of users' ratings of movies or items and ndarray of randomly chosen numbers
    """
    np.random.seed(SEED)
    chosen = np.random.choice(upper_limit, SAMPLE_SIZE_FOR_N)
    # Sorting to make populating X easier later
    chosen.sort()
    X = np.empty(shape=(SAMPLE_SIZE_FOR_N, num_users), dtype="uint8")
    return X, chosen


def load_netflix():
    """
    Reads 4 Netflix data .txt files and transforms the data into a numpy ndarray of movies by users,
    with each cell representing a rating

    :return: numpy ndarray of movies by users
    """
    X, movies = initialize_sampling_necessities(NUM_NETFLIX_MOVIES, NUM_NETFLIX_USERS)

    # For loop iterates through 4 data files
    for idx in range(NUM_DATA_FILES):
        raw_fp = os.path.join("raw_data", "combined_data_" + str(idx + 1) + ".txt")

        with open(raw_fp, "r") as fin:
            # represents the mapping between a movie_id and a number between 0 and 9999
            movie_idx = None
            # says if current customer row is for a randomly chosen movie
            chosen_movie = False
            for line in fin:
                # if this line declares a movie
                if ":" in line:
                    # Gets the movie id and subtracts 1 since it is 1-indexed
                    movie_id = int(line.split(":")[0]) - 1
                    # Returns the index at which movie_id appears in movies if found, otherwise empty ndarray
                    movie_idx = np.where(movies == movie_id)[0]
                    # If movie is randomly chosen one
                    if len(movie_idx) != 0:
                        chosen_movie = True
                        # Gets numerical value of the index
                        movie_idx = movie_idx[0]
                    else:
                        chosen_movie = False
                # else this line declares a customer
                else:
                    if chosen_movie:
                        user_row_parts = line.split(",")
                        # Subtracts 1 since customer id is 1-indexed
                        cust_id = int(user_row_parts[0]) - 1
                        rating = int(user_row_parts[1])
                        X[movie_idx][cust_id] = rating

    np.save("netflix_dataset_array", X)


def load_yahoo():
    """
    Reads Yahoo! Music data .txt file and transforms the data into a numpy ndarray of items by users, with each cell representing a rating

    :return: numpy ndarray of items by users
    """
    X, items = initialize_sampling_necessities(NUM_YAHOO_ITEMS, NUM_YAHOO_USERS)
    raw_fp = os.path.join("raw_data", "trainIdx1.txt")

    with open(raw_fp, "r") as fin:
        # holds the current user id
        user_id = None
        for line in fin:
            # if this line declares a user
            if "|" in line:
                # Gets the user id
                user_id = int(line.split("|")[0])
            # else this line declares an item
            else:
                item_row_parts = line.split("\t")
                item_id = int(item_row_parts[0])
                # Returns the index at which item_id appears in items if found, otherwise empty ndarray
                item_idx = np.where(items == item_id)[0]
                # If item is randomly chosen one
                if len(item_idx) != 0:
                    rating = int(item_row_parts[1])
                    # item_idx[0] gets numerical value of the index
                    X[item_idx[0]][user_id] = rating

    np.save("yahoo_dataset_array", X)
