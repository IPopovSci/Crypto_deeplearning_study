from sklearn.model_selection import train_test_split
import pandas as pd
from Arguments import args

'''Accepts a pandas dataframe, reads the size of train and test sizes from the arguments dictionary, and uses sklearn library
implementation for splitting the dataframe into 2 portions'''

def train_test_split_custom(df):
    train_size = args['train_size']
    test_size = args['test_size']

    df_train, df_test = train_test_split(df, train_size, test_size, shuffle=False)

    return df_train,df_test

