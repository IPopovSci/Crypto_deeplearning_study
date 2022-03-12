import os

import tensorflow as tf
from dotenv import load_dotenv
from tensorflow.keras.callbacks import ModelCheckpoint
from Networks.data_params.index import *

from Arguments import args
from Data_Processing.data_trim import trim_dataset
from Networks.structures.Conv1DLSTM import create_convlstm_model
from Networks.structures.Ensemble_model import create_model_ensembly
from pipeline import data_prep
# from plotting import plot_results
from training.callbacks import ResetStatesOnEpochEnd
import numpy as np

load_dotenv()

'''Module for training new models'''
ticker = args['ticker']
MM_path = os.getenv('MM_Path')
SS_path = os.getenv('SS_Path')

#x_t, y_t, x_val, y_val, x_test_t, y_test_t = data_prep('pancake',initial_training=True,batch=False,SS_path = 'F:\MM\scalers\\bnbusdt_ss_pancake1min',MM_path = 'F:\MM\scalers\\bnbusdt_mm_pancake1min',big_update=False)
BATCH_SIZE = args['batch_size']

'''Singular Model training function'''
#money = np.full(shape=BATCH_SIZE,fill_value=1000)
def train_model(ensembly = False):
    x_t, y_t, x_val, y_val, x_test_t, y_test_t,size = data_prep('pancake',ta=True,initial_training=True,batch=False,SS_path = SS_path,MM_path = MM_path,big_update=False)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode='min', patience=100)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.85,
                                                     patience=8, min_lr=0.000000000001,
                                                     verbose=1, mode='min')
    reset_states = ResetStatesOnEpochEnd()
    mcp = ModelCheckpoint(
        os.path.join(f'F:\MM\models\\bnbusdt\\1min\\',
                     "{val_loss:.8f}_{val_metric_signs:.8f}-best_model-{epoch:02d}.h5"),
        monitor='val_loss', verbose=3,
        save_best_only=False, save_weights_only=False, mode='min', period=1)
    if ensembly == False:
        lstm_model = create_convlstm_model(x_t)
    else:
        lstm_model = create_model_ensembly(x_t)


    history_lstm = lstm_model.fit(x=[trim_dataset(x_t, BATCH_SIZE),trim_dataset(y_t, BATCH_SIZE)],y=trim_dataset(y_t, BATCH_SIZE), epochs=10000,
                                  verbose=1, batch_size=BATCH_SIZE,
                                  shuffle=False, validation_data=([trim_dataset(x_val, BATCH_SIZE),trim_dataset(y_val,BATCH_SIZE)],
                                                                  trim_dataset(y_val, BATCH_SIZE)),
                                  callbacks=[mcp, reduce_lr,early_stop])


train_model(ensembly=False)

