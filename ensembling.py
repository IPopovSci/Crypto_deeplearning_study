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

ticker = 'GME'
x_t, y_t, x_val, y_val, x_test_t, y_test_t = data_prep(ticker)
BATCH_SIZE = args['batch_size']
epoch = None
val_loss = None


def train_models(x_t, y_t, x_val, y_val, num_models=10, model_name='Default'):
    for i in range(num_models):
        lstm_model = create_model(x_t)
        tf.keras.backend.clear_session()
        mcp = ModelCheckpoint(
            os.path.join(f'data\output\models\{model_name}', "best_model-{epoch:02d}-{val_loss:.4f}.h5"),
            monitor='val_loss', verbose=2,
            save_best_only=True, save_weights_only=False, mode='min', period=1)

        history_lstm = lstm_model.fit(x_t, y_t, epochs=args["epochs"], verbose=1, batch_size=BATCH_SIZE,
                                      shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                                                                      trim_dataset(y_val, BATCH_SIZE)), callbacks=[mcp])


# train_models(x_t,y_t,x_val,y_val,5,'LSTM_MSFT')

def simple_mean_ensemble(ticker, model_name='Default',update=True):
    preds = []
    x_t, y_t, x_val, y_val, x_test_t, y_test_t = data_prep(ticker)

    for model in os.listdir(f'data\output\models\{model_name}'):
        saved_model = load_model(os.path.join(f'data\output\models\{model_name}', model),
                                 custom_objects={'custom_loss': custom_loss, 'attention': Attention})
        if update == True:
            saved_model.reset_states()
            history_lstm = saved_model.fit(x_t, y_t, epochs=args["epochs"], verbose=1, batch_size=BATCH_SIZE,
                                           shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                                                                           trim_dataset(y_val, BATCH_SIZE)))
        y_pred_lstm = saved_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
        y_pred_lstm = y_pred_lstm.flatten()
        y_pred, y_test = unscale_data(ticker, y_pred_lstm, y_test_t)
        preds.append(y_pred)

    mean_preds = np.mean(preds,axis=0)
    print(mean_preds)

    y_test = trim_dataset(y_test, BATCH_SIZE)
    plot_results(mean_preds,y_test)

def update_models(ticker_list=['AMD', 'AXR', 'TSLA','DIS','GME','^GSPC','AAPL'], model_name_load='Default',
                  model_name_save='Default'):
    for model in os.listdir(f'data\output\models\{model_name_load}'):
        saved_model = load_model(os.path.join(f'data\output\models\{model_name_load}', model),
                                 custom_objects={'custom_loss': custom_loss, 'attention': Attention})
        i = 0

        for ticker in ticker_list:
            x_t, y_t, x_val, y_val, x_test_t, y_test_t = data_prep(ticker)

            mcp = ModelCheckpoint(
                os.path.join(f'data\output\models\{model_name_save}\{model}\{i}',
                             "best_model-{epoch:02d}-{val_loss:.4f}.h5"),
                monitor='val_loss', verbose=2,
                save_best_only=True, save_weights_only=False, mode='min', period=1)

            history_lstm = saved_model.fit(x_t, y_t, epochs=args["epochs"], verbose=1, batch_size=BATCH_SIZE,
                                           shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                                                                           trim_dataset(y_val, BATCH_SIZE)),
                                           callbacks=[mcp])
            i+=1

simple_mean_ensemble(ticker,model_name='the_best',update=True)
# update_models(model_name_load='LSTM_MSFT', model_name_save='Multi_Update_Models')
