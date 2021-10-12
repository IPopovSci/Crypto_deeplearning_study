from sklearn.model_selection import train_test_split
import pandas as pd
from Arguments import args
import numpy as np

'''Accepts a pandas dataframe, reads the size of train and test sizes from the arguments dictionary, and uses sklearn library
implementation for splitting the dataframe into 2 portions'''


def train_test_split_custom(df):
    train_size = args['train_size']
    test_size = args['test_size']

    df_train, df_test = train_test_split(df, train_size=train_size, test_size=test_size, shuffle=False)

    df_validation, df_test = train_test_split(df_test, train_size=0.5, test_size=0.5)

    return df_train, df_validation, df_test


'''This is a split between x (data) and y (targets). The amount of targets is currently hardcoded - something to think about in the future
Since we have 5 targets, first 5 columns of original database represent those targets, while the rest is input data'''
'''If we are using the custom loss in custom_function, we won't need to shift x and y variables, but we will have to do it if we decide not to use that loss'''



def x_y_split(x_train, x_validation, x_test):
    x_train, x_validation, x_test = np.nan_to_num(x_train), np.nan_to_num(x_validation), np.nan_to_num(
        x_test)  # Get rid of any potential NaN values

    y_train = x_train[:,
              :5]  # Separate the target variables (5 because we have 5 target variables, which are the first 5 columns of original dataset)
    x_train = x_train[:, 5:]  # Get the x data

    y_test = x_test[:, :5]
    x_test = x_test[:, 5:]

    y_validation = x_validation[:, :5]
    x_validation = x_validation[:, 5:]

    return x_train, x_validation, x_test, y_train, y_validation, y_test
