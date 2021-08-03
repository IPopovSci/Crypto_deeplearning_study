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
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale,scale
from keras.preprocessing.sequence import TimeseriesGenerator

ticker = args['ticker']

'''Step 1 - Download stock price data from yahoo finance'''
ticker_data(ticker)
'''Step 3 - Add TA Analysis'''
add_ta(ticker)
'''Step 4'''
'''Loading the dataset and converting it into price differences'''
df = pd.read_csv(f'data/02_ta/{ticker}.csv')

df = np.array(df)
df = np.delete(df, 0, 1)  # This will remove the date field
# df = np.delete(df,0,1) #This will remove VIX - Why is it empty?

df_differences = data_proc.row_difference(df) #This will populate an empty numpy array with differences in values;Last column is the price column - I double checked
x_t = df_differences

y_t =  x_t[:, -1:] #This will grab only the last column, which si prices
x_t = x_t[:, :-1] #Grabbing all columns but the last one, X data


"""Step 5"""
'''Scaling data so PCA has easier time working with it'''
''''''
x_t = scale(x_t)  # This, reportedly, scales one feature at a time
y_t = scale(y_t)

'''Step 5.5 - use PCA reduction to reduce number of features (Faster training, less feature burden on the model'''
np.nan_to_num(x_t,copy=False) #get rid of nans
pca = PCA(n_components=0.99,svd_solver = 'full')
x_t = pca.fit_transform(x_t)

'''Step 5.7'''
'''Min_max everything to between 0 and 1, then multiply by 255 to get the pixel value'''
x_t = minmax_scale(x_t, feature_range=(0, 1))  # This, reportedly, scales one feature at a time
x_t = (x_t * 255).astype(np.uint8)
y_t = (y_t * 255).astype(np.uint8)
print(y_t)
for row in range(y_t.shape[0]):
    y_t[row:,:] = y_t[row+1:,:]

print(y_t)
'''Step 6g'''
"""Split the data into training/testing"""
from sklearn.model_selection import train_test_split

last_col = x_t.shape[1]
X_train, X_test, Y_train, Y_test = train_test_split(
    x_t, y_t, test_size=0.2,shuffle=False) #Slicing into training and test - check that the slice is correct

'''Step 7'''
'''The step where we try keras timeseries generator even though we have a perfectly fine working of our own'''
timeseries_dataset_from_array
'''Step N - profits'''
w, h = X_train[1], X_train[0]  # Declared the Width and Height of an Image
#print(X_train.shape[1],h)
t = (h, w, 1)  # To store pixels
# Creation of Array
i = Image.fromarray(X_train)
i.show()
#i.save('^IXIC_X_TRAIN_GIVEITTOYA.png')
