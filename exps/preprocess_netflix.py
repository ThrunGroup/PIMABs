import pandas as pd
import numpy as np
import math
import re
import os
import time
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.model_selection import cross_validate

start_time = time.time()
filename = os.path.dirname(os.path.dirname(__file__))
for file in ["data", "netflix", "combined_data_1.txt", "combined_data_1.txt"]:
    filename = os.path.join(filename, file)
df = pd.read_csv(filename, header=None, names=['Cust_Id', 'Rating'], usecols = [0,1])

df['Rating'] = df['Rating'].astype(float)

print('Dataset 1 shape: {}'.format(df.shape))
print('-Dataset examples-')
print(df.iloc[:100, :])
print("data loading: ", time.time() - start_time)

df_nan = pd.DataFrame(pd.isnull(df.Rating))
df_nan = df_nan[df_nan['Rating']==True]
df_nan = df_nan.reset_index()

movie_np = []
movie_id = 1

for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
    # numpy approach
    temp = np.full((1, i-j-1), movie_id)
    movie_np = np.append(movie_np, temp)
    movie_id += 1

# Account for last record and corresponding length
# numpy approach
last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1), movie_id)
movie_np = np.append(movie_np, last_record)

print('Movie numpy: {}'.format(movie_np))
print('Length: {}'.format(len(movie_np)))
print(time.time() - start_time)

print("data cleaning: ", time.time() - start_time)

# remove those Movie ID rows
df = df[pd.notnull(df['Rating'])]

df['Movie_Id'] = movie_np.astype(int)
df['Cust_Id'] = df['Cust_Id'].astype(int)
print('-Dataset examples-')
print(df.iloc[::5000000, :])

f = ['count','mean']

df_movie_summary = df.groupby('Movie_Id')['Rating'].agg(f)
df_movie_summary.index = df_movie_summary.index.map(int)
movie_benchmark = round(df_movie_summary['count'].quantile(0.7),0)
drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

print('Movie minimum times of review: {}'.format(movie_benchmark))

df_cust_summary = df.groupby('Cust_Id')['Rating'].agg(f)
df_cust_summary.index = df_cust_summary.index.map(int)
cust_benchmark = round(df_cust_summary['count'].quantile(0.7),0)
drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

print('Customer minimum times of review: {}'.format(cust_benchmark))

print('Original Shape: {}'.format(df.shape))
df = df[~df['Movie_Id'].isin(drop_movie_list)]
df = df[~df['Cust_Id'].isin(drop_cust_list)]
print('After Trim Shape: {}'.format(df.shape))
print('-Data Examples-')
print(df.iloc[::5000000, :])
df.to_csv("netflix_cleaned1")

# df_p = pd.pivot_table(df, values='Rating', index='Cust_Id', columns='Movie_Id')
# print(df_p.shape)
#
# df_p.to_csv("netflix_table1")
df = pd.read_csv("netflix_cleaned1")
reader = Reader()
data = Dataset.load_from_df(df[['Movie_Id', 'Cust_Id', 'Rating']][:], reader)
trainset = data.build_full_trainset()
model = SVD(n_factors=20)
print("start fitting")
model.fit(trainset=trainset)
print(model.pu.shape)
print(model.qi.shape)
np.save("Movie_factors_20_new", model.pu)
np.save("Movie_biases_20_new", model.bu)
np.save("Customer_factors_20_new", model.qi)
np.save("Customer_biases_20_new", model.bi)
