from yfinance_facade import ticker_data
from plotting import plot_stock
from ta_feature_add import add_ta
from PCA import pca_reduction
from tt_split import train_test_split_custom
from data_scaling import SS_transform,min_max_sc
from Arguments import args
from build_timeseries import build_timeseries
from data_trim import trim_dataset
from LSTM_network import create_lstm_model
import numpy as np
import tensorflow as tf

ticker = '^GSPC'

BATCH_SIZE = args['batch_size']

def data_prep(ticker):
    '''Step 1 - Download stock price data from yahoo finance'''
    ticker_data(ticker)
    '''Step 2 - Plot stock price & volume'''
    # plot_stock(ticker, False)
    '''Step 3 - Add TA Analysis'''
    add_ta(ticker)
    '''Step 4 - Split data into training set and test set'''
    train_test_split_custom(ticker)
    '''Step 5 - Perform StanardScaler Reduction'''
    SS_transform(ticker)
    '''Step 6 - Perform PCA Reduction'''
    pca_reduction(ticker)
    '''Step 6 - Perform MinMaxScaling'''
    x_train, x_test = min_max_sc(ticker)
    '''Step 7 - Create Time-series'''
    y_col_index = args['n_components'] - 1  # Minus one because y_col_index searches for the next column (I.e have to indicate the previous one)
    x_t, y_t = build_timeseries(x_train, y_col_index)
    x_t = trim_dataset(x_t, BATCH_SIZE)
    y_t = trim_dataset(y_t, BATCH_SIZE)
    '''Step 8 - Initialize Model'''
    #lstm_model = create_lstm_model(x_t)
    '''Step 9 - Break Test into test and validation'''
    x_temp, y_temp = build_timeseries(x_test, y_col_index)
    x_val, x_test_t = np.array_split(trim_dataset(x_temp, BATCH_SIZE), 2)
    y_val, y_test_t = np.array_split(trim_dataset(y_temp, BATCH_SIZE), 2)
    print("Test size", x_test_t.shape, y_test_t.shape, x_val.shape, y_val.shape)
    return x_t,y_t,x_val,y_val,x_test_t,y_test_t

def create_model(x_t):
    lstm_model = create_lstm_model(x_t)
    return lstm_model

