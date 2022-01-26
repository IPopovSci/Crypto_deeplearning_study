import pandas as pd
from Data_Processing.get_data import scv_data
from Arguments import args
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import numpy as np
from Arguments import args
tf.config.experimental_run_functions_eagerly(True)
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
import joblib

def one_to_five(data):
    ohlc = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    df = data.resample('5min', base=0).apply(ohlc)
    df.dropna(inplace=True)

    print(df)

# ticker = args['ticker']
# history = scv_data(ticker)
# #print(history.head())
# one_to_five(history)
# def continue_learning(ticker, model):
#     saved_model = load_model(f'F:\MM\models_update\{ticker}\{model}.h5',
#                              custom_objects={'SeqSelfAttention': SeqSelfAttention,
#                                              'mean_squared_error_custom': mean_squared_error_custom})
#     early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=16)
#
#     saved_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#                         loss=mean_squared_error_custom)
#
#     reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
#                                                      patience=6, min_lr=0.0000000001,
#                                                      verbose=1, mode='min')
#
#     x_t, y_t = data_prep_transfer('cryptowatch')
#
#     mcp = ModelCheckpoint(
#         os.path.join(f'F:\MM\models_update\\bnbusdt_save\\',
#                      "{loss:.8f}-best_model-{epoch:02d}.h5"),
#         monitor='loss', verbose=3,
#         save_best_only=False, save_weights_only=False, mode='min', period=1)
#
#     history_lstm = saved_model.fit(trim_dataset(x_t, BATCH_SIZE), trim_dataset(y_t, BATCH_SIZE), epochs=256, verbose=1,
#                                    batch_size=BATCH_SIZE,
#                                    shuffle=False,
#                                    callbacks=[mcp, early_stop, reduce_lr])
#     # saved_model.reset_states()

import pandas as pd
import glob
import os

def join_files():
    joined_files = os.path.join("F:\MM\Data\BNBUSDT", "bnbusdt*.csv")

    joined_list = glob.glob(joined_files)

    for file in joined_list:
        df = pd.read_csv(file)

        print(df)
        print(file)

        df.set_index('time_period_start',inplace=True)
        print(df)
        print(file)

    f = pd.concat(map(pd.read_csv, joined_list), ignore_index=False,axis=0,join='outer')
    f.drop_duplicates(ignore_index=False,inplace=True)

    f.sort_values(by='time_period_start', ascending=1,inplace=True)
    #df.drop(columns='Unnamed: 0',inplace=True)
    f.set_index('time_period_start', inplace=True)
    f.to_csv('F:\MM\Data\BNBUSDT\\bnbusdt_merge.csv',index=True)

#def multiply_volume_by_price():

'''Rewriting inverse transform in tensorflow notation to use in loss function'''

import inspect


MM_path = 'F:\MM\scalers\BNBusdt_MM'
SS_path = 'F:\MM\scalers\BNBusdt_SS'

mm_y = joblib.load(MM_path + ".y")
sc_y = joblib.load(SS_path + ".y")
attributes = inspect.getmembers(mm_y, lambda a:not(inspect.isroutine(a)))
attributes = ([a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]) #
# print(attributes)
# print(attributes[5])
# print(attributes[8])

def tf_mm_inverse_transform(X):
    MM_path = 'F:\MM\scalers\BNBusdt_MM'
    SS_path = 'F:\MM\scalers\BNBusdt_SS'

    mm_y = joblib.load(MM_path + ".y")
    sc_y = joblib.load(SS_path + ".y")
    #X = ops.convert_to_tensor_v2(X,dtype=tf.float32)

    X = (X - K.constant(mm_y.min_)) / K.constant(mm_y.scale_)

    return X