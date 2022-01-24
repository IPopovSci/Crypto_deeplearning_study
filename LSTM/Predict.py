from pipeline import data_prep,data_prep_batch_1,data_prep_transfer
from Arguments import args
from Data_Processing.data_trim import trim_dataset
from tensorflow.keras.callbacks import ModelCheckpoint
from LSTM.callbacks import custom_loss,ratio_loss,my_metric_fn,mean_squared_error_custom
from tensorflow.keras.models import load_model
from attention import Attention
#from plotting import plot_results
from Old_and_crap.LSTM_Model_Ensembly import create_model_ensembly_average
import numpy as np
import os
from keras_self_attention import SeqSelfAttention
from Backtesting.Backtest_DaysCorrect import backtest
import tensorflow as tf

BATCH_SIZE = args['batch_size']
ticker = 'bnbusdt'
def predict(ticker, model_name='Default',update=False,load_weights='False'):
    preds = []
    back_test_info = []
    x_t, y_t = data_prep_transfer('cryptowatch')

    print(x_t)
    saved_model = load_model(os.path.join(f'F:\MM\models\\bnbusdt\\5m_transfer', f'{model_name}.h5'),
                             custom_objects={'SeqSelfAttention': SeqSelfAttention,'mean_squared_error_custom':mean_squared_error_custom})
    saved_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
                        loss=mean_squared_error_custom)
    if update == True:
        history_lstm = saved_model.fit(trim_dataset(x_t, BATCH_SIZE), trim_dataset(y_t, BATCH_SIZE),
                                       epochs=1, verbose=1, batch_size=BATCH_SIZE,
                                       shuffle=False)

    y_pred_lstm = saved_model.predict(trim_dataset(x_t, BATCH_SIZE), batch_size=BATCH_SIZE)
    #y_pred_lstm = y_pred_lstm.flatten()
    # y_pred, y_test = unscale_data(ticker, y_pred_lstm, y_test_t)

    print('Model',y_pred_lstm)
    #backtest(y_test_t, y_pred_lstm)

predict(ticker,'0.08364866_0.02244842-best_model-01_125000',True)