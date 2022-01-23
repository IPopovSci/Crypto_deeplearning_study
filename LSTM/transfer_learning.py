from pipeline import data_prep,data_prep_transfer,data_prep_batch_1,data_prep_batch_2
from Arguments import args
from Data_Processing.data_trim import trim_dataset
from tensorflow.keras.callbacks import ModelCheckpoint
from LSTM.callbacks import custom_loss,ratio_loss,my_metric_fn,mean_squared_error_custom
from tensorflow.keras.models import load_model
from keras_self_attention import SeqSelfAttention
import numpy as np
import os
import tensorflow as tf
from LSTM.LSTM_network_transfer import create_lstm_model_transfer

BATCH_SIZE = args['batch_size']
ticker = 'bnbusdt'


def transfer_learning(ticker, model,start,increment):
    x_t, y_t, x_val, y_val, x_test_t, y_test_t, size = data_prep_batch_1('CSV')

    end = increment + start

    x_t, y_t = data_prep_batch_2(x_t, y_t, start, end)

    lstm_model = create_lstm_model_transfer(x_t,model)



    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.0005,
        decay_steps=100,
        decay_rate=0.99,
        staircase=True)

    mcp = ModelCheckpoint(
        os.path.join(f'F:\MM\models\\bnbusdt\\',
                     "{val_loss:.8f}_{loss:.8f}-best_model-{epoch:02d}.h5"),
        monitor='val_loss', verbose=3,
        save_best_only=False, save_weights_only=False, mode='min', period=1)



    end = increment + start

    x_t, y_t, x_val, y_val, x_test_t, y_test_t, size = data_prep_batch_1('CSV')
    while end < size:
        x_train, y_train = data_prep_batch_2(x_t, y_t, start, end)
        history_lstm = lstm_model.fit(trim_dataset(x_train, BATCH_SIZE), trim_dataset(y_train, BATCH_SIZE),
                                       epochs=1,
                                       verbose=1, batch_size=BATCH_SIZE,
                                       shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                                                                       trim_dataset(y_val, BATCH_SIZE)),
                                       callbacks=[mcp])
        start = end
        end += increment

transfer_learning('bnbusdt','0.00030713_0.00702284-best_model-01',0,50000)