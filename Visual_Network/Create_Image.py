import pandas as pd
from yfinance_facade import ticker_data
from Arguments import args
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from Arguments import args
import joblib
import numpy as np
import os
from ta_feature_add import add_ta
from Visual_Network import data_proc
from PIL import Image
from matplotlib import pyplot as plt
ticker = args['ticker']

'''Step 1 - Download stock price data from yahoo finance'''
ticker_data(ticker)
'''Step 3 - Add TA Analysis'''
add_ta(ticker)
'''Step 4'''
'''Loading the dataset and converting it into price differences'''
df = pd.read_csv(f'data/02_ta/{ticker}.csv')

df = np.array(df)
df = np.delete(df,0,1) #This will remove the date field
df = np.delete(df,0,1) #This will remove VIX - Why is it empty?

df_differences = data_proc.row_difference(df)
len_x_t = df_differences.shape[1] #All the columns except the last one
print(len_x_t)
y_t = df_differences[:,-1:] #This will grab all the price closures, it's the last column of the array (Check that numpy didnt re-ordered them)
x_t = df_differences[:,:len_x_t - 1]#Double check this later

"""Step 5"""
'''Min/Max values to 0-255 to represent pixel value'''
sc = MinMaxScaler((0,1))
y_t = sc.fit_transform(y_t)
x_t = sc.fit_transform(x_t)

x_t = (x_t * 255).astype(np.uint8)
y_t = (y_t * 255).astype(np.uint8)
# x_t = x_t.astype(int)
# y_t = y_t.astype(int)
print(x_t)
"""Testing"""
#x_t = x_t[0:88,:]
print(x_t.shape)

'''Step N - profits'''
w,h=x_t[1],x_t[0]     # Declared the Width and Height of an Image
t=(h,w,1)       # To store pixels
# Creation of Array
i=Image.fromarray(y_t)
i.show()
i.save('^IXIC_y.png')



