import numpy as np
import time


movie_factors = np.load("Movie_factors_20_new.npy")
customer_factors = np.load("Customer_factors_20_new.npy")
movie_biases = np.load("Movie_biases_20_new.npy")
customer_biases = np.load("Customer_biases_20_new.npy")

print(movie_factors.shape, customer_factors.shape)

start_time = time.time()
data = (movie_factors + np.expand_dims(movie_biases, axis=1)) @ \
       ((customer_factors + np.expand_dims(customer_biases, axis=1)).transpose())

np.save("Movie_ratings", data)

