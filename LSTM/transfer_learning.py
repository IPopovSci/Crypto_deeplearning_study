from pipeline import data_prep
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
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=16)

        saved_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                      loss=mean_squared_error_custom,
                      metrics=my_metric_fn)

        x_t, y_t, x_val, y_val, x_test_t, y_test_t = data_prep('cryptowatch')
        x_total = np.concatenate((x_t, x_val,x_test_t))
        y_total = np.concatenate((y_t, y_val,y_test_t))
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=3, min_lr=0.000000000000000000000000000000000001,verbose=1,mode='min')
        mcp = ModelCheckpoint(
            os.path.join(f'F:\MM\models\{ticker}\\',
                         "{val_loss:.8f}_{loss:.8f}-best_model-{epoch:02d}.h5"),
            monitor='val_loss', verbose=3,
            save_best_only=False, save_weights_only=False, mode='min', period=1)

        history_lstm = saved_model.fit(trim_dataset(x_total,BATCH_SIZE),trim_dataset(y_total,BATCH_SIZE), epochs=32, verbose=1, batch_size=BATCH_SIZE,
                                       shuffle=False, validation_data=(trim_dataset(x_test_t, BATCH_SIZE),
                                                                       trim_dataset(y_test_t, BATCH_SIZE)),
                                       callbacks=[mcp,early_stop,reduce_lr])
        #saved_model.reset_states()
update_model(ticker,'test')