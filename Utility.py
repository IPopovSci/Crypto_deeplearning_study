import pandas as pd
from Data_Processing.get_data import scv_data
from Arguments import args

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

def join_files():
    joined_files = os.path.join("F:\MM\Data\BNBUSDT", "bnbusdt*.csv")

    joined_list = glob.glob(joined_files)

    for file in joined_list:
        df = pd.read_csv(file)

        print(df)
        print(file)

        df.set_index('time_period_start',inplace=True)
        print(df)
        print(file)

    f = pd.concat(map(pd.read_csv, joined_list), ignore_index=False,axis=0,join='outer')
    f.drop_duplicates(ignore_index=False,inplace=True)

    f.sort_values(by='time_period_start', ascending=1,inplace=True)
    #df.drop(columns='Unnamed: 0',inplace=True)
    f.set_index('time_period_start', inplace=True)
    f.to_csv('F:\MM\Data\BNBUSDT\\bnbusdt_merge.csv',index=True)

join_files()
