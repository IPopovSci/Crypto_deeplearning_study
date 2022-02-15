from pipeline import data_prep
from Arguments import args
from Data_Processing.data_trim import trim_dataset
from tensorflow.keras.callbacks import ModelCheckpoint
# from plotting import plot_results
from LSTM.callbacks import ResetStatesOnEpochEnd
import numpy as np
import os
import tensorflow as tf
from LSTM.LSTM_network import create_lstm_model as create_model

'''Module for training new models'''
ticker = args['ticker']

#x_t, y_t, x_val, y_val, x_test_t, y_test_t = data_prep('pancake',initial_training=True,batch=False,SS_path = 'F:\MM\scalers\\bnbusdt_ss_pancake1min',MM_path = 'F:\MM\scalers\\bnbusdt_mm_pancake1min',big_update=False)
BATCH_SIZE = args['batch_size']

'''Singular Model training function'''

def train_model():
    x_t, y_t, x_val, y_val, x_test_t, y_test_t,size = data_prep('testing',ta=False,initial_training=True,batch=False,SS_path = 'F:\MM\scalers\\bnbusdt_ss_pancake1min',MM_path = 'F:\MM\scalers\\bnbusdt_mm_pancake1min',big_update=False)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='metric_signs', factor=0.2,
                                                     patience=4, min_lr=0.000000000001,
                                                     verbose=1, mode='max')
    reset_states = ResetStatesOnEpochEnd()
    mcp = ModelCheckpoint(
        os.path.join(f'F:\MM\models\\bnbusdt\\1min\\',
                     "{val_loss:.8f}_{val_metric_signs:.8f}-best_model-{epoch:02d}.h5"),
        monitor='val_loss', verbose=3,
        save_best_only=False, save_weights_only=False, mode='min', period=1)

    lstm_model = create_model(x_t)
    x_total = np.concatenate((x_t, x_val))
    y_total = np.concatenate((y_t, y_val))


    history_lstm = lstm_model.fit(trim_dataset(x_t, BATCH_SIZE), trim_dataset(y_t, BATCH_SIZE), epochs=1000,
                                  verbose=1, batch_size=BATCH_SIZE,
                                  shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                                                                  trim_dataset(y_val, BATCH_SIZE)),
                                  callbacks=[mcp, reduce_lr,reset_states])


train_model()

