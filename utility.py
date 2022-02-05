
#import tensorflow as tf
import tensorflow.keras.backend as K


#tf.config.experimental_run_functions_eagerly(True)

import joblib


def one_to_five(data):
    ohlc = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    df = data.resample('5min', base=0).apply(ohlc)
    df.dropna(inplace=True)

    print(df)


# ticker = args['ticker']
# history = scv_data(ticker)
# #print(history.head())
# one_to_five(history)
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

import pandas as pd
import glob
import os


def join_files(path_load, path_save):
    joined_files = os.path.join(f"{path_load}", "bnb*.csv")

    joined_list = glob.glob(joined_files)

    for file in joined_list: #this whole loop can/should be avoided
        df = pd.read_csv(file)

        #df['time'] = pd.to_datetime(df['time'], unit='s').dt.strftime('%Y-%m-%dT%H:%M')
        #df['time'] = df['time'].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%dT%H:%M'))
        try:
            df.rename(
                columns={'time_period_end': 'time', 'price_open': 'Open', 'price_high': 'High', 'price_low': 'Low',
                         'price_close': 'Close', 'volume_traded': 'Volume'},
                inplace=True)
            df.rename(
                columns={'time_period_start': 'time', 'price_open': 'Open', 'price_high': 'High', 'price_low': 'Low',
                         'price_close': 'Close', 'volume_traded': 'Volume'},
                inplace=True)
        except:
            print("no need to rename")

        col = ['time', 'Open', 'High', 'Low', 'Close',
               'Volume']
        df = df[col]

        df['time'] = pd.to_datetime(df['time'], infer_datetime_format=True,format='%Y-%m-%dT%H:%M',utc=True)

        df.set_index('time', inplace=True)

        df.to_csv(file)



    f = pd.concat(map(pd.read_csv, joined_list), ignore_index=False, axis=0, join='outer')


    f.sort_values(by='time', ascending=1, inplace=True)
    f.drop_duplicates(ignore_index=False, inplace=True, subset=['time'])

    # df.drop(columns='Unnamed: 0',inplace=True)
    f.set_index('time', inplace=True)

    f.to_csv(f'{path_save}\\bnbusdt_pancake.csv', index=True)

    return f


# join_files()

# def multiply_volume_by_price():


def tf_mm_inverse_transform(X):
    MM_path = 'F:\MM\scalers\BNBusdt_MM'
    SS_path = 'F:\MM\scalers\BNBusdt_SS'

    mm_y = joblib.load(MM_path + ".y")
    sc_y = joblib.load(SS_path + ".y")
    # X = ops.convert_to_tensor_v2(X,dtype=tf.float32)

    X = (X - K.constant(mm_y.min_)) / K.constant(mm_y.scale_)

    return X
