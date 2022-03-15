import pandas as pd
import glob
import os
import tensorflow.keras.backend as K
from dotenv import load_dotenv
from pipeline.pipelineargs import PipelineArgs
from Networks.network_config import NetworkParams

import joblib

load_dotenv()
pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()
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

def unscale(y_true,y_pred):
    mm_y = joblib.load(pipeline_args.args['mm_y_path'])
    sc_y = joblib.load(pipeline_args.args['ss_y_path'])

    y_true_un = (((y_true - K.constant(mm_y.min_)) / K.constant(mm_y.scale_)) * K.constant(sc_y.scale_)) + K.constant(
        sc_y.mean_)

    y_pred_un = (((y_pred - K.constant(mm_y.min_)) / K.constant(mm_y.scale_)) * K.constant(sc_y.scale_)) + K.constant(
        sc_y.mean_)

    return y_true_un,y_pred_un

def structure_create():
    #Check and create folders for scalers if they don't exist
    if not os.path.exists(os.getenv('ss_path') + f'\{pipeline_args.args["interval"]}' + f'\\{pipeline_args.args["ticker"]}'):
        os.makedirs(os.getenv('ss_path') + f'\{pipeline_args.args["interval"]}' + f'\\{pipeline_args.args["ticker"]}', mode=0o777)

    if not os.path.exists(os.getenv('mm_path') + f'\{pipeline_args.args["interval"]}' + f'\\{pipeline_args.args["ticker"]}'):
        os.makedirs(os.getenv('mm_path') + f'\{pipeline_args.args["interval"]}' + f'\\{pipeline_args.args["ticker"]}', mode=0o777)

    #Check and create folders for model saving

    if not os.path.exists(os.getenv('model_path') + f'\{pipeline_args.args["interval"]}\{pipeline_args.args["ticker"]}\{network_args.network["model_type"]}'):
        os.makedirs(os.getenv('model_path') + f'\{pipeline_args.args["interval"]}\{pipeline_args.args["ticker"]}\{network_args.network["model_type"]}', mode=0o777)

def hash_folder_create():
    if not os.path.exists(os.getenv('model_path') + f'\{pipeline_args.args["interval"]}\{pipeline_args.args["ticker"]}\{network_args.network["model_type"]}'):
        os.makedirs(os.getenv('model_path') + f'\{pipeline_args.args["interval"]}\{pipeline_args.args["ticker"]}\{network_args.network["model_type"]}', mode=0o777)


