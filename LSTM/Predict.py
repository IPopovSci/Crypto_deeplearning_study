from pipeline import data_prep,data_prep_transfer,data_prep_batch_2
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
from numpy import mean

BATCH_SIZE = args['batch_size']
ticker = 'bnbusdt'
def predict(ticker, model_name='Default',update=False,load_weights='False'):
    preds = []
    back_test_info = []
    x_t, y_t, x_val, y_val, x_test_t, y_test_t,size = data_prep('CSV',initial_training=False,batch=True)

    saved_model = load_model(os.path.join(f'F:\MM\models\\bnbusdt\\', f'{model_name}.h5'),
                             custom_objects={'SeqSelfAttention': SeqSelfAttention,'custom_cosine_similarity':custom_cosine_similarity,'mean_squared_error_custom':mean_squared_error_custom})
    saved_model.compile(optimizer=tf.keras.optimizers.Adagrad(),
                        loss= custom_cosine_similarity)
    saved_model.reset_states()
    #x_ts, y_ts = data_prep_batch_2(x_t, y_t, 0, 1500)
    if update == True:
        history_lstm = saved_model.fit(trim_dataset(x_test_t[-100:], BATCH_SIZE), trim_dataset(y_test_t[-100:], BATCH_SIZE),
                                       epochs=1, verbose=1, batch_size=BATCH_SIZE,
                                       shuffle=False)

    #y_pred_lstm = saved_model(trim_dataset(x_test_t[-1000:], BATCH_SIZE),training=False)


    y_pred_lstm = saved_model.predict(trim_dataset(x_test_t[-2000:-1000], BATCH_SIZE), batch_size=BATCH_SIZE)
    y_pred_lstm = saved_model.predict(trim_dataset(x_test_t[-1000:], BATCH_SIZE), batch_size=BATCH_SIZE)
    print('pred done')
    # y_pred_lstm = y_pred_lstm.reshape(-1,1)
    # y_pred_lstm = y_pred_lstm.flatten()

    #print("Loss before inverse",custom_cosine_similarity(y_test_t[-50:-1,:],y_pred_lstm[-50:-1,:]))
    #print('Preds before inverse',y_pred_lstm[-25:-1,:])
    # MM_path = 'F:\MM\scalers\BNBusdt_MM'
    # SS_path = 'F:\MM\scalers\BNBusdt_SS'
    # mm_y = joblib.load(MM_path + ".y")
    # sc_y = joblib.load(SS_path + ".y")
    # y_test_t = mm_y.inverse_transform(y_test_t)
    # y_test_t = sc_y.inverse_transform(y_test_t)
    #
    # y_pred_lstm = mm_y.inverse_transform(y_pred_lstm)
    #
    # y_pred_lstm = sc_y.inverse_transform(y_pred_lstm)

    print('test after inversing:',y_test_t[-25:,:])
    # print('------------------------------------')
    #
    print('Model',y_pred_lstm[-25:,:])
    #print("Loss after inverse",mean_squared_error_custom(y_test_t[-50:-1,:], y_pred_lstm[-50:-1,:]))
    #backtest(y_test_t, y_pred_lstm)

predict(ticker,'3864.14477539_7452.23876953-best_model-01',False)

#pred: -------+---+----------+
#true: -++-++-+++-+-+----+----
