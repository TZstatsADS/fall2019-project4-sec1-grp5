import pandas as pd
import ipdb
import numpy as np
df =  pd.read_csv("../data/ml-latest-small/movies.csv")
df['movie_index'] = [i for i in range(9742)]
movie_index2id = dict(zip(df.movieId, df.movie_index))


df2 = pd.read_csv("bi_data1.csv")
new = []
for i in range(len(df2['movieId'])):
	update = movie_index2id[df2['movieId'][i]]
	new.append(update)
df2['movie_index'] = new
bibin = df2[['movie_index','bi']].to_numpy()

movie_bias_time = np.zeros((9742))

for i in range(len(bibin)):
	movie_bias_time[int(bibin[i,0])] = bibin[i,1]


ipdb.set_trace()