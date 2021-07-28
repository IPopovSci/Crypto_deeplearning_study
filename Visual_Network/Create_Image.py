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
import png


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

scaler_x = np.full((x_t.shape[0],x_t.shape[1]), 255, dtype = int)
scaler_y = np.full((y_t.shape[0],y_t.shape[1]), 255, dtype = int)
x_t = scaler_x*x_t #Multiple
y_t = scaler_y*y_t
x_t = np.round(x_t) #This rounds the numbers to ints and we should be done!
y_t = np.round(y_t)
x_t = x_t.astype(int)
y_t = y_t.astype(int)
print(x_t)


'''Step N - profits'''
with open('/Visual_Network/foo_gray.png', 'wb') as f:
    writer = png.Writer(width=x_t.shape[1], height=x_t.shape[0], bitdepth=16, greyscale=True)
    writer.write(f, x_t)

# w = png.Writer(x_t.shape[1], x_t.shape[0], greyscale=True, bitdepth=15)
# f = open('test.png', 'wb')
# w.write(f, x_t)
# png.from_array(x_t,'L').save("/tmp/foo.png")




