from run_functions import data_prep, create_model
from Arguments import args
from data_trim import trim_dataset
from keras.callbacks import ModelCheckpoint
from callbacks import mcp, custom_loss
from keras.models import Sequential, load_model
from attention import Attention
from data_scaling import unscale_data,unscale_data_np
from plotting import plot_results
import numpy as np
import os
import tensorflow as tf
import random
from Backtesting import up_or_down,back_test

ticker = args['ticker']

x_t, y_t, x_val, y_val, x_test_t, y_test_t = data_prep(ticker)
BATCH_SIZE = args['batch_size']
epoch = None
val_loss = None


def train_models(x_t, y_t, x_val, y_val, num_models=1, model_name='Default',multiple=False):
    continuous_list = ['RSV=F','BTC-USD','AAPL','AMZN','MSFT','DIS','PG','KO','PFE','^DJI','^RUT','^GSPC','^IXIC']
    for i in range(num_models):
        lstm_model = create_model(x_t)
        tf.keras.backend.clear_session()
        if multiple:
            for ticker in continuous_list:
                mcp = ModelCheckpoint(
                    os.path.join(f'data\output\models\{model_name}',
                                 "7step-bigbrain_{val_loss}.h5".format(i=i, val_loss='{val_loss:.4f}')),
                    monitor='val_loss', verbose=2,
                    save_best_only=True, save_weights_only=False, mode='min', period=1)
                print(f'Now Training {ticker}')
                x_t, y_t, x_val, y_val, x_test_t, y_test_t = data_prep(ticker)
                history_lstm = lstm_model.fit(x_t, y_t, epochs=args["epochs"], verbose=1, batch_size=BATCH_SIZE,
                                              shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                                                                              trim_dataset(y_val, BATCH_SIZE)),
                                              callbacks=[mcp])

                #lstm_model.reset_states()
            else:
                history_lstm = lstm_model.fit(x_t, y_t, epochs=args["epochs"], verbose=1, batch_size=BATCH_SIZE,
                                            shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                                                                            trim_dataset(y_val, BATCH_SIZE)), callbacks=[mcp])


#train_models(x_t,y_t,x_val,y_val,25,'NASDAQ_1_Step_Multitrain_1LSTM_10000',multiple=True)

def simple_mean_ensemble(ticker, model_name='Default',update=True):
    preds = []
    back_test_info = []
    x_t, y_t, x_val, y_val, x_test_t, y_test_t = data_prep(ticker)
    x_total = np.concatenate((x_t, x_val))
    y_total = np.concatenate((y_t, y_val))

    for model in os.listdir(f'data\output\models\{model_name}'):
        if model.endswith('.h5'):
            saved_model = load_model(os.path.join(f'data\output\models\{model_name}', model),
                                     custom_objects={'custom_loss': custom_loss, 'attention': Attention})
            if update == True:
                history_lstm = saved_model.fit(trim_dataset(x_val, BATCH_SIZE), trim_dataset(y_val, BATCH_SIZE),
                                               epochs=4, verbose=1, batch_size=BATCH_SIZE,
                                               shuffle=False, validation_data=(trim_dataset(x_test_t, BATCH_SIZE),
                                                                               trim_dataset(y_test_t, BATCH_SIZE)))
            y_pred_lstm = saved_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
            y_pred_lstm = y_pred_lstm.flatten()
            y_pred, y_test = unscale_data(ticker, y_pred_lstm, y_test_t)
            preds.append(y_pred)
            print('Model',model)
            back_test(y_pred, y_test)


    mean_preds = np.mean(preds,axis=0)

    y_test = trim_dataset(y_test, BATCH_SIZE)
    up_or_down(mean_preds)
    back_test(mean_preds,y_test)

    plot_results(mean_preds,y_test)

def update_models(ticker_list=['^IXIC'], model_name_load='Default',
                  model_name_save='Default'):
    for model in os.listdir(f'data\output\models\{model_name_load}'):
        saved_model = load_model(os.path.join(f'data\output\models\{model_name_load}', model),
                                 custom_objects={'custom_loss': custom_loss, 'attention': Attention})

        i = 0

        for ticker in ticker_list:
            x_t, y_t, x_val, y_val, x_test_t, y_test_t = data_prep(ticker)
            print(type(x_t))
            x_total = np.concatenate((x_t,x_val))
            y_total = np.concatenate((y_t,y_val))
            mcp = ModelCheckpoint(
                os.path.join(f'data\output\models\{model_name_save}\{model}\\',
                             "best_model-{epoch:02d}-{val_loss:.4f}.h5"),
                monitor='val_loss', verbose=3,
                save_best_only=True, save_weights_only=False, mode='min', period=1)

            history_lstm = saved_model.fit(trim_dataset(x_val,BATCH_SIZE),trim_dataset(y_val,BATCH_SIZE), epochs=1, verbose=1, batch_size=BATCH_SIZE,
                                           shuffle=False, validation_data=(trim_dataset(x_test_t, BATCH_SIZE),
                                                                           trim_dataset(y_test_t, BATCH_SIZE)),
                                           callbacks=[mcp])
            saved_model.reset_states()
            i+=1

simple_mean_ensemble(ticker,model_name='working_models\\NASDAQ_1_Step_Multitrain_1LSTM_10000_Update',update=True)
#update_models(model_name_load='NASDAQ_1_Step_Multitrain_1LSTM_10000', model_name_save='NASDAQ_1_Step_Multitrain_1LSTM_10000_Update')
