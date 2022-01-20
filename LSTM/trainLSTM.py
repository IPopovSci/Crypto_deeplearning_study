from pipeline import data_prep
from Arguments import args
from Data_Processing.data_trim import trim_dataset
from tensorflow.keras.callbacks import ModelCheckpoint
from LSTM.callbacks import custom_loss, ratio_loss, my_metric_fn, mean_squared_error_custom
from tensorflow.keras.models import load_model
from attention import Attention
# from plotting import plot_results
from LSTM.LSTM_Model_Ensembly import create_model_ensembly_average
import numpy as np
import os
import tensorflow as tf
import random
from Backtesting_old import up_or_down, back_test
import statistics
from LSTM.LSTM_network import create_lstm_model as create_model
from keras_self_attention import SeqSelfAttention
from Backtesting.Backtest_DaysCorrect import backtest

'''Module for training new models'''
ticker = args['ticker']

x_t, y_t, x_val, y_val, x_test_t, y_test_t = data_prep('CSV')
BATCH_SIZE = args['batch_size']

'''Singular Model training function'''

def train_model(x_t, y_t, x_val, y_val, x_test_t, y_test_t, model_name='Default'):
    x_t, y_t, x_val, y_val, x_test_t, y_test_t = data_prep('CSV')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                     patience=4, min_lr=0.000000000000000000000000000000000001,
                                                     verbose=1, mode='min')
    mcp = ModelCheckpoint(
        os.path.join(f'data\output\\',
                     "{val_loss:.8f}_{loss:.8f}-best_model-{epoch:02d}.h5"),
        monitor='val_loss', verbose=3,
        save_best_only=False, save_weights_only=False, mode='min', period=1)

    lstm_model = create_model(x_t)
    x_total = np.concatenate((x_t, x_val))
    y_total = np.concatenate((y_t, y_val))
    history_lstm = lstm_model.fit(trim_dataset(x_total, BATCH_SIZE), trim_dataset(y_total, BATCH_SIZE), epochs=256,
                                  verbose=1, batch_size=BATCH_SIZE,
                                  shuffle=False, validation_data=(trim_dataset(x_test_t, BATCH_SIZE),
                                                                  trim_dataset(y_test_t, BATCH_SIZE)),
                                  callbacks=[mcp, reduce_lr])


train_model(x_t, y_t, x_val, y_val, x_test_t, y_test_t, 'ethusd')

def train_model_batch(model_name='Default'):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                     patience=4, min_lr=0.000000000000000000000000000000000001,
                                                     verbose=1, mode='min')
    mcp = ModelCheckpoint(
        os.path.join(f'F:\models\{ticker}\\',
                     "{val_loss:.8f}_{loss:.8f}-best_model-{epoch:02d}.h5"),
        monitor='val_loss', verbose=3,
        save_best_only=False, save_weights_only=False, mode='min', period=1)

    lstm_model = create_model(x_t)
    x_total = np.concatenate((x_t, x_val))
    y_total = np.concatenate((y_t, y_val))
    history_lstm = lstm_model.fit(trim_dataset(x_total, BATCH_SIZE), trim_dataset(y_total, BATCH_SIZE), epochs=256,
                                  verbose=1, batch_size=BATCH_SIZE,
                                  shuffle=False, validation_data=(trim_dataset(x_test_t, BATCH_SIZE),
                                                                  trim_dataset(y_test_t, BATCH_SIZE)),
                                  callbacks=[mcp, reduce_lr])