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
from build_timeseries import build_timeseries_conv

ticker = args['ticker']

'''Step 1 - Download stock price data from yahoo finance'''
ticker_data(ticker)
'''Step 2 - Add TA Analysis'''
add_ta(ticker)
'''Step 3'''
'''Loading the dataset and converting it into price differences'''
df = pd.read_csv(f'data/02_ta/{ticker}.csv')

df = np.array(df) #Convert into numpy array
df = np.delete(df, 0, 1)  # This will remove the date field
# df = np.delete(df,0,1) #This will remove VIX - Why is it empty? (Its only empty for the first n rows)

x_t = data_proc.row_difference(df) #This will populate an empty numpy array with differences in values;Last column is the price column - I double checked
"""Step 4"""
'''Scaling data so PCA has easier time working with it'''
''''''
x_t = scale(x_t)  # This, reportedly, scales one feature at a time
y_t = x_t[:,-1:] #This is only the last column I.e prices
x_t = x_t[:,:-1] #We don't want to PCA out y values

'''Step 5 - use PCA reduction to reduce number of features (Faster training, less feature burden on the model'''
np.nan_to_num(x_t,copy=False) #get rid of nans
pca = PCA(n_components=0.99,svd_solver = 'full') #This will create a PCA object to apply to our x data
x_t = pca.fit_transform(x_t) #Applying to x data



'''Step 6'''
'''Min_max everything to between 0 and 1, then multiply by 255 to get the pixel value'''
print(x_t.shape)
x_t = np.concatenate((x_t,y_t),axis=1) #Now that we skipped PCA for y values, we can return the array to having the price in it

x_t = minmax_scale(x_t, feature_range=(0, 1))  # This, reportedly, scales one feature at a time
x_t = (x_t * 255).astype(np.uint8) #Converting everything to grayscale values of 0-1 * 255
'''Step 7'''
'''Build Timeseries out of data'''
x_t= build_timeseries_conv(x_t) #This will create sliding windows, add one extra channel dimension, as well as transpose to bring it to the right dimension order - (Num_steps, height, width, channels)
#print('x_t',x_t)
'''Step 8'''
"""Split the data into training/testing"""
indexes = np.arange(x_t.shape[0])
np.random.shuffle(indexes)
train_index = indexes[: int(0.9 * x_t.shape[0])]
val_index = indexes[int(0.9 * x_t.shape[0]) :]
train_dataset = x_t[train_index]
val_dataset = x_t[val_index]

'''Step 9'''
''' We'll define a helper function to shift the frames, where
 `x` is frames 0 to n - 1, and `y` is frames 1 to n.'''

def create_shifted_frames(data):
    x = data[:, 0 : data.shape[1] - 1, :, :]
    y = data[:, 1 : data.shape[1], :, :]
    return x, y

# Apply the processing function to the datasets.
x_train, y_train = create_shifted_frames(train_dataset)
x_val, y_val = create_shifted_frames(val_dataset)

'''Step N - profits'''
'''Code for turning stuff into jpeg, if needed'''
# w, h = x_t[2], x_t[1]  # Declared the Width and Height of an Image
# #print(X_train.shape[1],h)
# t = (h, w, 1)  # To store pixels
# # Creation of Array
# i = Image.fromarray(x_t)
# i.show()
# #i.save('^IXIC_X_TRAIN_GIVEITTOYA.png')
