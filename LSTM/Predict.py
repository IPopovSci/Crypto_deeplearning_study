from pipeline import data_prep,data_prep_transfer
from Arguments import args
from Data_Processing.data_trim import trim_dataset
from tensorflow.keras.callbacks import ModelCheckpoint
from LSTM.callbacks import custom_loss,ratio_loss,my_metric_fn,mean_squared_error_custom,custom_cosine_similarity
from tensorflow.keras.models import load_model
from attention import Attention
#from plotting import plot_results
from Old_and_crap.LSTM_Model_Ensembly import create_model_ensembly_average
import numpy as np
import os
from keras_self_attention import SeqSelfAttention
from Backtesting.Backtest_DaysCorrect import backtest
import tensorflow as tf
import joblib

BATCH_SIZE = args['batch_size']
ticker = 'bnbusdt'
def predict(ticker, model_name='Default',update=False,load_weights='False'):
    preds = []
    back_test_info = []
    x_t, y_t, x_val, y_val, x_test_t, y_test_t,size,SS_scaler,mm_scaler = data_prep('CSV',initial_training=False,batch=True,SS_path='F:\MM\scalers\BNBusdt_SS',MM_path='F:\MM\scalers\BNBusdt_MM')

    saved_model = load_model(os.path.join(f'F:\MM\models\\bnbusdt\\', f'{model_name}.h5'),
                             custom_objects={'SeqSelfAttention': SeqSelfAttention,'custom_cosine_similarity':custom_cosine_similarity})
    saved_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
                        loss=mean_squared_error_custom)
    if update == True:
        history_lstm = saved_model.fit(trim_dataset(x_val[-1000:-1,:], BATCH_SIZE), trim_dataset(y_val[-1000:-1,:], BATCH_SIZE),
                                       epochs=1, verbose=1, batch_size=BATCH_SIZE, validation_data=(trim_dataset(x_test_t, BATCH_SIZE),
                                                                      trim_dataset(y_test_t, BATCH_SIZE)),
                                       shuffle=False)

    y_pred_lstm = saved_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
    # y_pred_lstm = y_pred_lstm.reshape(-1,1)
    # y_pred_lstm = y_pred_lstm.flatten()

    print("Loss before inverse",custom_cosine_similarity(y_test_t,y_pred_lstm))
    MM_path = 'F:\MM\scalers\BNBusdt_MM'
    SS_path = 'F:\MM\scalers\BNBusdt_SS'
    mm_y = joblib.load(MM_path + ".y")
    sc_y = joblib.load(SS_path + ".y")
    y_test_t = mm_y.inverse_transform(y_test_t)
    y_test_t = sc_y.inverse_transform(y_test_t)

    y_pred_lstm = mm_y.inverse_transform(y_pred_lstm)

    y_pred_lstm = sc_y.inverse_transform(y_pred_lstm)

    print('test after inversing:',y_test_t[-100:-1,:])
    # print('------------------------------------')
    #
    print('Model',y_pred_lstm[-100:-1,:])
    print("Loss after inverse", custom_cosine_similarity(y_test_t, y_pred_lstm))
    #backtest(y_test_t, y_pred_lstm)

predict(ticker,'0.96980858_0.97656322-best_model-01',False)

