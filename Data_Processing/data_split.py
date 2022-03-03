from sklearn.model_selection import train_test_split
import pandas as pd
from Arguments import args
import numpy as np

'''Accepts a pandas dataframe, reads the size of train and test sizes from the arguments dictionary, and uses sklearn library
implementation for splitting the dataframe into 2 portions
Outputs 3 dataframes for train,validation and test'''


def train_test_split_custom(df):
    train_size = args['train_size']
    test_size = args['test_size']

    df_train, df_test = train_test_split(df, train_size=train_size, test_size=test_size, shuffle=False,random_state=42)

    df_validation, df_test = train_test_split(df_test, train_size=0.5, test_size=0.5,random_state=42,shuffle=False)

    return df_train, df_validation, df_test



'''Function for splitting inputs into inputs and outputs
Accepts training, validation and testing data
y_type is a switch: "close" will grab only the closing price, 'ohclv' will grab OHCLV data, 'testing' grabs only the first feature'''


def x_y_split(x_train, x_validation, x_test,y_type = 'close'):

    '''converting pandas> numpy'''
    x_train = x_train.to_numpy()
    x_validation = x_validation.to_numpy()
    x_test = x_test.to_numpy()



    x_train, x_validation, x_test = np.nan_to_num(x_train), np.nan_to_num(x_validation), np.nan_to_num(
        x_test)  # Get rid of any potential NaN values

    if y_type == 'close':
        y_train = x_train[:,
                  3]
        y_test = x_test[:, 3]
        y_validation = x_validation[:, 3]
    elif y_type == 'ohlcv':
        y_train = x_train[:,
                  :5]
        y_test = x_test[:, :5]
        y_validation = x_validation[:, :5]
    elif y_type == 'testing':
        y_train = x_train[:,
                  0]
        y_test = x_test[:, 0]
        y_validation = x_validation[:, 0]

    y_train_t,y_validation_t,y_test_t = y_train.reshape(-1,1),y_validation.reshape(-1,1),y_test.reshape(-1,1) #This reshaping is needed for SS
    print(y_train_t.shape,y_validation_t.shape,y_test_t.shape)

    return x_train, x_validation, x_test, y_train_t, y_validation_t, y_test_t

'''x_y split for singular database'''

def x_y_split_small(x_train):
    x_train = np.nan_to_num(x_train)  # Get rid of any potential NaN values

    y_train = x_train[:,
              :5]  # Separate the target variables (5 because we have 5 target variables, which are the first 5 columns of original dataset)
    x_train = x_train[:, 5:]  # Get the x data


    return x_train,y_train
