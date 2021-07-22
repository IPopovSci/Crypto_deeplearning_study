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
    continuous_list = ['^GSPC','^DJI','^RUT','^IXIC']
    for i in range(num_models):
        lstm_model = create_model(x_t)
        tf.keras.backend.clear_session()
        mcp = ModelCheckpoint(
            os.path.join(f'data\output\models\{model_name}', f"3step-continuous{i}.h5"),
            monitor='val_loss', verbose=2,
            save_best_only=True, save_weights_only=False, mode='min', period=1)
        if multiple:
            for ticker in continuous_list:
                print(f'Now Training {ticker}')
                x_t, y_t, x_val, y_val, x_test_t, y_test_t = data_prep(ticker)
                history_lstm = lstm_model.fit(x_t, y_t, epochs=args["epochs"], verbose=1, batch_size=BATCH_SIZE,
                                              shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                                                                              trim_dataset(y_val, BATCH_SIZE)),
                                              callbacks=[mcp])

                lstm_model.reset_states()
        else:
            history_lstm = lstm_model.fit(x_t, y_t, epochs=args["epochs"], verbose=1, batch_size=BATCH_SIZE,
                                        shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                                                                        trim_dataset(y_val, BATCH_SIZE)), callbacks=[mcp])


train_models(x_t,y_t,x_val,y_val,15,'NASDAQ',multiple=True)

def simple_mean_ensemble(ticker, model_name='Default',update=True):
    preds = []
    x_t, y_t, x_val, y_val, x_test_t, y_test_t = data_prep(ticker)

    for model in os.listdir(f'data\output\models\{model_name}'):
        saved_model = load_model(os.path.join(f'data\output\models\{model_name}', model),
                                 custom_objects={'custom_loss': custom_loss, 'attention': Attention})
        if update == True:
            saved_model.reset_states()
            history_lstm = saved_model.fit(trim_dataset(x_val, BATCH_SIZE), trim_dataset(y_val, BATCH_SIZE), epochs=15, verbose=1, batch_size=BATCH_SIZE,
                                           shuffle=False)
        y_pred_lstm = saved_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
        y_pred_lstm = y_pred_lstm.flatten()
        y_pred, y_test = unscale_data(ticker, y_pred_lstm, y_test_t)
        preds.append(y_pred)


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
        saved_model.reset_states()
        i = 0

        for ticker in ticker_list:
            x_t, y_t, x_val, y_val, x_test_t, y_test_t = data_prep(ticker)
            print(type(x_t))
            x_total = np.concatenate((x_t,x_val))
            y_total = np.concatenate((y_t,y_val))
            mcp = ModelCheckpoint(
                os.path.join(f'data\output\models\{model_name_save}\{model}\{i}',
                             "best_model-{epoch:02d}-{val_loss:.4f}.h5"),
                monitor='val_loss', verbose=2,
                save_best_only=True, save_weights_only=False, mode='min', period=1)

            history_lstm = saved_model.fit(x_t,y_t, epochs=10, verbose=1, batch_size=BATCH_SIZE,
                                           shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                                                                           trim_dataset(y_val, BATCH_SIZE)),
                                           callbacks=[mcp])
            i+=1

#simple_mean_ensemble(ticker,model_name='NASDAQ_best_7step',update=False)
# update_models(model_name_load='NASDAQ', model_name_save='NASDAQ_Update')
