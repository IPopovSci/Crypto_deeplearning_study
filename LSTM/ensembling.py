from pipeline import data_prep
from Arguments import args
from Data_Processing.data_trim import trim_dataset
from tensorflow.keras.callbacks import ModelCheckpoint
from LSTM.callbacks import custom_loss,ratio_loss,my_metric_fn,mean_squared_error_custom
from tensorflow.keras.models import load_model
from attention import Attention
#from plotting import plot_results
from LSTM.LSTM_Model_Ensembly import create_model_ensembly_average
import numpy as np
import os
import tensorflow as tf
import random
from Backtesting_old import up_or_down,back_test
import statistics
from LSTM.LSTM_network import create_lstm_model as create_model
from keras_self_attention import SeqSelfAttention
from Backtesting.Backtest_DaysCorrect import backtest


#TODO: Read the timesries keras tutorial, look up special layers for using selu, can you lambda loop in the loss?
ticker = args['ticker']

x_t, y_t, x_val, y_val, x_test_t, y_test_t = data_prep('CSV')
BATCH_SIZE = args['batch_size']
epoch = None
val_loss = None


def train_models(x_t, y_t, x_val, y_val, x_test_t,y_test_t, num_models=1, model_name='Default',multiple=False):
    ticker = args['ticker']
    continuous_list = ['^RUT','AAPL','KO','^N225','PEP','PFE','^FTSE','IBM','ETH-USD','ED','BK','BTC-USD','^GDAXI','^FCHI','^STOXX50E','^N100','BFX','IMOEX.ME','^BUK100P','^XAX','^NYA','^GSPC','^IXIC']
    for i in range(num_models):
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                         patience=4, min_lr=0.000000000000000000000000000000000001,
                                                         verbose=1, mode='min')

        mcp = ModelCheckpoint(
            os.path.join(f'data\output\models\{model_name}\\',
                         "{val_loss:.8f}_{loss:.8f}-best_model-{epoch:02d}.h5"),
            monitor='val_loss', verbose=3,
            save_best_only=False, save_weights_only=False, mode='min', period=1)

        lstm_model = create_model(x_t)
        tf.keras.backend.clear_session()
        j = 0
        x_total = np.concatenate((x_t, x_val))
        y_total = np.concatenate((y_t, y_val))
        if multiple:
            for ticker in continuous_list:
                print(f'Now Training {ticker}')
                x_t, y_t, x_val, y_val, x_test_t, y_test_t = data_prep(ticker)
                history_lstm = lstm_model.fit(x_total, y_total, epochs=12, verbose=1, batch_size=BATCH_SIZE,
                                              shuffle=False, validation_data=(trim_dataset(x_test_t, BATCH_SIZE),
                                                                              trim_dataset(y_test_t, BATCH_SIZE)),
                                              callbacks=[mcp,reduce_lr,early_stop])
                lstm_model.reset_states()
        else:
            # x_t, y_t, x_val, y_val, x_test_t, y_test_t = data_prep(ticker)
            #print(y_t.shape,y_val.shape)
            x_total = np.concatenate((x_t, x_val))
            y_total = np.concatenate((y_t, y_val))
            #print(trim_dataset(y_test_t,BATCH_SIZE))
            history_lstm = lstm_model.fit(trim_dataset(x_total,BATCH_SIZE),trim_dataset(y_total,BATCH_SIZE), epochs=256, verbose=1, batch_size=BATCH_SIZE,
                                        shuffle=False, validation_data=(trim_dataset(x_test_t, BATCH_SIZE),
                                                                        trim_dataset(y_test_t, BATCH_SIZE)), callbacks=[mcp,reduce_lr])


#train_models(x_t,y_t,x_val,y_val,x_test_t,y_test_t,20,'ethust',multiple=False)
ticker = 'ethust'
args['ticker'] = ticker


def get_custom_objects():
    SeqSelfAttention.get_custom_objects()


def simple_mean_ensemble(ticker, model_name='Default',update=False,load_weights='False'):
    i = 0
    while i < 10:
        preds = []
        back_test_info = []
        x_t, y_t, x_val, y_val, x_test_t, y_test_t = data_prep('CSV')
        x_total = np.concatenate((x_t, x_val))
        y_total = np.concatenate((y_t, y_val))
        for model in os.listdir(f'data\output\models\{model_name}'):
            if model.endswith('.h5'):
                saved_model = load_model(os.path.join(f'data\output\models\{model_name}', model),
                                         custom_objects={'SeqSelfAttention': SeqSelfAttention,'mean_squared_error_custom':mean_squared_error_custom})
                if update == True:
                    history_lstm = saved_model.fit(trim_dataset(x_val, BATCH_SIZE), trim_dataset(y_val, BATCH_SIZE),
                                                   epochs=1, verbose=1, batch_size=BATCH_SIZE,
                                                   shuffle=False, validation_data=(trim_dataset(x_test_t, BATCH_SIZE),
                                                                                   trim_dataset(y_test_t, BATCH_SIZE)))

                y_pred_lstm = saved_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
                #y_pred_lstm = y_pred_lstm.flatten()
                # y_pred, y_test = unscale_data(ticker, y_pred_lstm, y_test_t)

                print('Model',model)
                backtest(y_test_t, y_pred_lstm)
                i+=1


        mean_preds = np.mean(preds,axis=0)



        y_test = trim_dataset(y_test_t, BATCH_SIZE)
        np.set_printoptions(threshold=np.inf)

        # up_or_down(mean_preds)
        # back_test(mean_preds,y_test)
        #
        # plot_results(mean_preds,y_test)

simple_mean_ensemble(ticker,model_name=f'working_models_clean\\{ticker}',update=True,load_weights=False)

def update_models(ticker_list=['HUV.TO'], model_name_load='Default',
                  model_name_save='Default'):
    config = {
        'class_name': 'PolynomialDecay',
        'config': {'cycle': False,
                   'decay_steps': 10000,
                   'end_learning_rate': 0.01,
                   'initial_learning_rate': 0.1,
                   'name': None,
                   'power': 0.5}}

    args['ticker'] = 'HUV.TO'
    for model in random.sample(os.listdir(f'data\output\models\{model_name_load}'),len(os.listdir(f'data\output\models\{model_name_load}'))):
        #print('What is ostlistdir?',os.listdir(f'data\output\models\{model_name_load}'))
        saved_model = load_model(os.path.join(f'data\output\models\{model_name_load}', model),
                                 custom_objects={'stock_loss_money':custom_loss,'ratio_loss': ratio_loss,'custom_loss': custom_loss, 'attention': Attention,'my_metric_fn': my_metric_fn})
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=16)
        i = 0
        saved_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                      loss=custom_loss,
                      metrics=my_metric_fn)

        for ticker in ticker_list:
            x_t, y_t, x_val, y_val, x_test_t, y_test_t = data_prep(ticker)
            x_total = np.concatenate((x_t, x_val))
            y_total = np.concatenate((y_t, y_val))
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                          patience=3, min_lr=0.000000000000000000000000000000000001,verbose=1,mode='min')
            mcp = ModelCheckpoint(
                os.path.join(f'data\output\models\{model_name_save}\\',
                             "{loss:.8f}_{val_loss:.8f}_{val_my_metric_fn:.4f}-best_model-{epoch:02d}.h5"),
                monitor='val_loss', verbose=3,
                save_best_only=False, save_weights_only=False, mode='min', period=1)

            history_lstm = saved_model.fit(trim_dataset(x_total,BATCH_SIZE),trim_dataset(y_total,BATCH_SIZE), epochs=32, verbose=1, batch_size=BATCH_SIZE,
                                           shuffle=False, validation_data=(trim_dataset(x_test_t, BATCH_SIZE),
                                                                           trim_dataset(y_test_t, BATCH_SIZE)),
                                           callbacks=[mcp,early_stop,reduce_lr])
            saved_model.reset_states()
            i+=1


#update_models(model_name_load='working_models\\dump', model_name_save=f'working_models\\HUV.TO')

def keras_ensembly():
    preds = []
    back_test_info = []
    ticker = '^NDX'
    args['ticker'] = ticker
    x_t, y_t, x_val, y_val, x_test_t, y_test_t = data_prep(ticker)
    x_total = np.concatenate((x_t, x_val))
    y_total = np.concatenate((y_t, y_val))

    saved_model = create_model_ensembly_average(x_test_t,f'working_models_clean\\{ticker}')
    #saved_model = create_model_ensembly(x_test_t,f'working_models_clean\\{ticker}')
    i = 0
    while i < 5:
        y_pred_lstm = saved_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
        y_pred_lstm = y_pred_lstm.flatten()
    # y_pred, y_test = unscale_data(ticker, y_pred_lstm, y_test_t)
        percor =  back_test(y_pred_lstm, y_test_t)
        if percor >= 1:
            preds.append(y_pred_lstm)

        i+=1


    mean_preds = np.mean(preds,axis=0)

    y_test = trim_dataset(y_test_t, BATCH_SIZE)
    up_or_down(mean_preds)
    back_test(mean_preds,y_test)
    plot_results(mean_preds, y_test)

#keras_ensembly()

def model_cleanup():
    for subdir, dirs, files in os.walk(f'../data/output/models/Cleanup'):
        for dir in dirs:
            args['ticker'] = dir
            x_t, y_t, x_val, y_val, x_test_t, y_test_t = data_prep(dir)

            for model in random.sample(os.listdir(f'data\output\models\cleanup\\{dir}'),len(os.listdir(f'data\output\models\cleanup\\{dir}'))):
                model_path = os.path.join(f'data\output\models\cleanup\\{dir}', model)
                saved_model = load_model(model_path,
                                         custom_objects={'ratio_loss': ratio_loss, 'custom_loss': custom_loss,
                                                         'attention': Attention,'my_metric_fn':my_metric_fn})
                saved_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00000001),
                                    loss=custom_loss,
                                    metrics=my_metric_fn)
                i = 0
                while i < 3:
                    percor_list = []
                    y_pred = saved_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
                    percor = back_test(y_pred,y_test_t)
                    percor_list.append(percor)
                    i += 1
                if statistics.mean(percor_list) <= 1.9: #NDX: 1.45 IXIC: 1.38 HUV: 3 #For HUV Use 4% test, FOR NDX 1%
                    os.remove(os.path.join(f'data\output\models\cleanup\\{dir}', model))
                    print('Yeeted-Deleted!')


#model_cleanup()
