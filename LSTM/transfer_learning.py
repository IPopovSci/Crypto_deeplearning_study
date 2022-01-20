from pipeline import data_prep,data_prep_transfer
from Arguments import args
from Data_Processing.data_trim import trim_dataset
from tensorflow.keras.callbacks import ModelCheckpoint
from LSTM.callbacks import custom_loss,ratio_loss,my_metric_fn,mean_squared_error_custom
from tensorflow.keras.models import load_model
from keras_self_attention import SeqSelfAttention
import numpy as np
import os
import tensorflow as tf


BATCH_SIZE = args['batch_size']
ticker = 'bnbusdt'
def update_model(ticker,model):
        saved_model = load_model(f'F:\MM\models_update\{ticker}\{model}.h5',
                                 custom_objects={'SeqSelfAttention': SeqSelfAttention,'mean_squared_error_custom':mean_squared_error_custom})
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=16)

        saved_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001),
                      loss=mean_squared_error_custom)

        x_t, y_t = data_prep_transfer('cryptowatch')

        # mcp = ModelCheckpoint(
        #     os.path.join(f'F:\MM\models\{ticker}\\',
        #                  "{val_loss:.8f}_{loss:.8f}-best_model-{epoch:02d}.h5"),
        #     monitor='val_loss', verbose=3,
        #     save_best_only=False, save_weights_only=False, mode='min', period=1)

        history_lstm = saved_model.fit(trim_dataset(x_t,BATCH_SIZE),trim_dataset(y_t,BATCH_SIZE), epochs=32, verbose=1, batch_size=BATCH_SIZE,
                                       shuffle=False,
                                       callbacks=[early_stop])
        #saved_model.reset_states()
update_model(ticker,'test')