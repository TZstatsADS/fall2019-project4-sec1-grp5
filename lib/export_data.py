import pandas as pd
from IPython.display import Image
import math
from datetime import datetime
from sklearn.model_selection import train_test_split
import numpy as np
import ipdb

df =  pd.read_csv("../data/ml-latest-small/ratings.csv")

df['time'] = df['timestamp'].apply(datetime.fromtimestamp)
df_time = df.sort_values('time', ascending = True)
df_time['time'] = pd.to_datetime(df_time.time).dt.strftime('%Y-%m-%d')


# 8, 8, 7 years
# split into bins
bin1 = df_time[df_time['time'] <= '2003-12-31']
bin2 = df_time[(df_time['time'] <= '2011-12-31') & (df_time['time'] > '2003-12-31')]
bin3 = df_time[(df_time['time'] > '2011-12-31' )]


bin1_train, bin1_test, r1_train, r1_test = train_test_split(bin1[['userId', 'movieId','time']], bin1['rating'], test_size=0.2, random_state=42)
bin2_train, bin2_test, r2_train, r2_test = train_test_split(bin2[['userId', 'movieId','time']], bin2['rating'], test_size=0.2, random_state=42)
bin3_train, bin3_test, r3_train, r3_test = train_test_split(bin3[['userId', 'movieId','time']], bin3['rating'], test_size=0.2, random_state=42)

bin1_train['rating'] = r1_train
bin2_train['rating'] = r2_train
bin3_train['rating'] = r3_train

bin1_test['rating'] = r1_test
bin2_test['rating'] = r2_test
bin3_test['rating'] = r3_test


# function to remove unseen movieId and userId and add them back to train dataset
def move_unseen(train, test):
    move1 = test[~test.movieId.isin(train.movieId)]
    test = test[test.movieId.isin(train.movieId)]
    
    move2 = test[~test.userId.isin(train.userId)]
    test = test[test.userId.isin(train.userId)]
    
    train = pd.concat([train, move1, move2])
    
    return train, test

bin1_train, bin1_test = move_unseen(bin1_train, bin1_test)
bin2_train, bin2_test = move_unseen(bin2_train, bin2_test)
bin3_train, bin3_test = move_unseen(bin3_train, bin3_test)

# convert to user-movie matrix
R1_train = bin1_train.pivot_table(index='userId', columns='movieId', values='rating')
R1_test = bin1_test.pivot_table(index='userId', columns='movieId', values='rating')

R2_train = bin2_train.pivot_table(index='userId', columns='movieId', values='rating')
R2_test = bin2_test.pivot_table(index='userId', columns='movieId', values='rating')

R3_train = bin3_train.pivot_table(index='userId', columns='movieId', values='rating')
R3_test = bin3_test.pivot_table(index='userId', columns='movieId', values='rating')

all_train = pd.concat([bin1_train, bin2_train, bin3_train])
R = all_train.pivot_table(index='userId', columns='movieId', values='rating') # all train item-user matrix

mu = all_train['rating'].mean()
### bu
bu = pd.DataFrame(np.nanmean(R, axis=1) - mu)  #bias for users among all train
bu['userId'] = R.index
bu.to_csv('bu_global.csv')

bu1 = bu[bu['userId'].isin(R1_train.index)]
bu2 = bu[bu['userId'].isin(R2_train.index)]
bu3 = bu[bu['userId'].isin(R3_train.index)]

### bi
bi = pd.DataFrame(np.nanmean(R, axis=0) - mu) #bias for movies among all train
bi['movieId'] = R.columns
#bi.to_csv('bi_global.csv')

bi1 = bi[bi['movieId'].isin(R1_train.columns)]
bi2 = bi[bi['movieId'].isin(R2_train.columns)]
bi3 = bi[bi['movieId'].isin(R3_train.columns)]

# calculate bi,bin(t)
mu1 = bin1_train['rating'].mean()
bit1 = pd.DataFrame(np.nanmean(R1_train, axis = 0) - mu1)
bi1 = pd.DataFrame(bi1.reset_index(drop= True)[0]+bit1[0])
bi1['movieId'] = R1_train.columns

mu2 = bin2_train['rating'].mean()
bit2 = pd.DataFrame(np.nanmean(R2_train, axis = 0) - mu2)
bi2 = pd.DataFrame(bi2.reset_index(drop= True)[0]+bit2[0])
bi2['movieId'] = R2_train.columns

mu3 = bin3_train['rating'].mean()
bit3 = pd.DataFrame(np.nanmean(R3_train, axis = 0) - mu3)
bi3 = pd.DataFrame(bi3.reset_index(drop= True)[0]+bit3[0])
bi3['movieId'] = R3_train.columns

ipdb.set_trace()
'''
df_list= [bin1_train,bin1_test,bin2_train,bin2_test,bin3_train,bin3_test]
df_names= ['bin1_train','bin1_test','bin2_train','bin2_test','bin3_train','bin3_test']

writer = pd.ExcelWriter('train_test_data.xlsx')
for i, df in enumerate(df_list):
    df.to_excel(writer,sheet_name=df_names[i])
writer.save() 
'''

