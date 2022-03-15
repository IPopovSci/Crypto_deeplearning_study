import pandas as pd
import glob
import os
import tensorflow.keras.backend as K
from dotenv import load_dotenv
from pipeline.pipelineargs import PipelineArgs
import joblib

load_dotenv()
pipeline_args = PipelineArgs.get_instance()
'''This function will resample all csv files in data_path\\interval_from folder to interval_to interval
accepts: interval_from - folder name with respective data interval
        interval_to - which interval to resample to, and which respective folder to save to'''
def resample(interval_from,interval_to):
    all_csv = os.path.join(os.getenv('data_path') + f'\{interval_from}','*.csv')

    all_csv_list = glob.glob(all_csv)

    for file in all_csv_list:
        df = pd.read_csv(file)

        filename = file.split('1min\\')[1] #Grab just the name of the csv for saving purposes

        df['time'] = pd.to_datetime(df['time'], unit='ms')  # Unix to datetime conversion

        ohlcv = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        df.set_index('time', inplace=True)

        df = df.resample(rule=f'{interval_to}', offset=0).apply(ohlcv)

        df.dropna(inplace=True)

        df.to_csv(os.getenv('data_path') + f'\{interval_to}' + f'\\{filename}')




#TODO: This function not needed anymore, since we will be using csv for training and cryptowatch for predictions
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

def unscale(y_true,y_pred):
    mm_y = joblib.load(pipeline_args.args['mm_y_path'])
    sc_y = joblib.load(pipeline_args.args['ss_y_path'])

    y_true_un = (((y_true - K.constant(mm_y.min_)) / K.constant(mm_y.scale_)) * K.constant(sc_y.scale_)) + K.constant(
        sc_y.mean_)

    y_pred_un = (((y_pred - K.constant(mm_y.min_)) / K.constant(mm_y.scale_)) * K.constant(sc_y.scale_)) + K.constant(
        sc_y.mean_)

    return y_true_un,y_pred_un