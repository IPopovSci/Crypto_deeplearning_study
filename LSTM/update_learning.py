from pipeline import data_prep,data_prep_transfer,data_prep_batch_2
from Arguments import args
from Data_Processing.data_trim import trim_dataset
from tensorflow.keras.callbacks import ModelCheckpoint
from LSTM.callbacks import custom_loss,ratio_loss,my_metric_fn,mean_squared_error_custom,custom_cosine_similarity,metric_signs
from tensorflow.keras.models import load_model
from keras_self_attention import SeqSelfAttention
import numpy as np
import os
import tensorflow as tf


BATCH_SIZE = args['batch_size']
ticker = 'bnbusdt'
def continue_learning_batch(ticker, model,start,increment,final_pass):
        x_t, y_t, x_val, y_val, x_test_t, y_test_t,size = data_prep('CSV',initial_training=False,batch=True,SS_path='F:\MM\scalers\BNBusdt_SS',MM_path='F:\MM\scalers\BNBusdt_MM')

        saved_model = load_model(f'F:\MM\models\{ticker}\\1min\{model}.h5',
                                 custom_objects={'SeqSelfAttention': SeqSelfAttention,'mean_squared_error_custom':mean_squared_error_custom,'custom_cosine_similarity':custom_cosine_similarity,'metric_signs':metric_signs})

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.00001,
            decay_steps=50,
            decay_rate=0.99,
            staircase=True)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_metric_signs', factor=0.5,
                                                         patience=3, min_lr=0.00000001,
                                                         verbose=1, mode='max')


        saved_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule,amsgrad=True,epsilon=0.1),
                      loss=custom_cosine_similarity,metrics=metric_signs)

        mcp = ModelCheckpoint(
            os.path.join(f'F:\MM\models\\{ticker}\\1min\\',
                         "{val_loss:.8f}_{val_metric_signs:.8f}-best_model-{epoch:02d}.h5"),
            monitor='val_loss', verbose=3,
            save_best_only=False, save_weights_only=False, mode='min', period=1)
        y_pred_lstm = saved_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)



        end = increment + start
        if end > size:
            end = increment
            start = 0
        while end < size:
            x_train, y_train = data_prep_batch_2(x_t, y_t, start, end)
            if final_pass == True:
                history_lstm = saved_model.fit(trim_dataset(x_val, BATCH_SIZE), trim_dataset(y_val, BATCH_SIZE),
                                               epochs=1,
                                               verbose=1, batch_size=BATCH_SIZE,
                                               shuffle=False, validation_data=(trim_dataset(x_test_t, BATCH_SIZE),
                                                                               trim_dataset(y_test_t, BATCH_SIZE)),
                                               callbacks=[mcp])
                history_lstm = saved_model.fit(trim_dataset(x_test_t, BATCH_SIZE), trim_dataset(y_test_t, BATCH_SIZE),
                                               epochs=1,
                                               verbose=1, batch_size=BATCH_SIZE,
                                               shuffle=False, validation_data=(trim_dataset(x_test_t, BATCH_SIZE),
                                                                               trim_dataset(y_test_t, BATCH_SIZE)),
                                               callbacks=[mcp])
            else:
                history_lstm = saved_model.fit(trim_dataset(x_train, BATCH_SIZE), trim_dataset(y_train, BATCH_SIZE),
                                              epochs=1,
                                              verbose=1, batch_size=BATCH_SIZE,
                                              shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                                                                              trim_dataset(y_val, BATCH_SIZE)),
                                              callbacks=[mcp,reduce_lr])
            start = end
            end += increment
            if end > size:
                end = increment
                start = 0

        #saved_model.reset_states()
continue_learning_batch(ticker, '0.32694387_55.34999847-best_model-01',0,25000,False)

#
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