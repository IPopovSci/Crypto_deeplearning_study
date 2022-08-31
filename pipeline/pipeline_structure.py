from Data_Processing.get_data import scv_data, cryptowatch_data,scv_data_to_sql,cryptowatch_data_update_database
from Data_Processing.ta_feature_add import add_ta
from Data_Processing.create_lags import lagged_returns
from Data_Processing.data_split import train_test_split_custom, x_y_split
from Data_Processing.data_scaling import SS_transform, min_max_transform
from Data_Processing.PCA import pca_reduction
from Data_Processing.build_timeseries import build_timeseries
from pipeline.pipelineargs import PipelineArgs
from dotenv import load_dotenv
from utility import structure_create
from sqlalchemy import create_engine
from pathlib import Path
import sys
import os
import pandas as pd

load_dotenv()
db_path = f'{Path(sys.path[0])}/sql/model_params.sqlite'  # Using SQLAlchemy to load data
print(db_path)
DB_NAME = "model_params.sqlite"
engine = create_engine(f'sqlite:///{db_path}', echo=False)
print(engine)

'''Pipeline function

This function obtains, augments, cleans and separates the data required for the neural network.

Outputs: train, validation, test arrays for inputs and outputs (x and y), as well as the size of training set.'''


def pipeline():
    '''Step 0: Create folder structure and grab settings + update database if asked to'''
    structure_create()
    pipeline_args = PipelineArgs.get_instance()

    if os.getenv('update_db_from_csv')=='yes':
        scv_data_to_sql(pipeline_args.args['ticker'], os.getenv('data_path'), pipeline_args.args['interval'],engine)
    if os.getenv('update_db_from_api')=='yes':
        cryptowatch_data_update_database(pipeline_args.args['ticker'], pipeline_args.args['interval'],engine)
    '''Step 1: Get Data'''
    history = pd.read_sql(f"SELECT * FROM {pipeline_args.args['ticker']}_{pipeline_args.args['interval']}_data",engine,index_col='time')

    # if pipeline_args.args['mode'] == 'training' or pipeline_args.args['mode'] == 'continue':
    #     history = scv_data(pipeline_args.args['ticker'], os.getenv('data_path'), pipeline_args.args['interval'])
    # elif pipeline_args.args['mode'] == 'prediction':
    #     history = cryptowatch_data(pipeline_args.args['ticker'], pipeline_args.args['interval'])
    #     history_r = history
    # else:
    #     print('Wrong mode! Currently supported modes: training,prediction,continue')
    '''Any additional feature extractors can go here'''
    print('pandas read sql')
    '''Step 2: Apply TA Analysis'''
    if pipeline_args.args['ta'] == True:
        history = add_ta(history)  # The columns names can be acessed from network_config 'train_cols'
    print('ta added')
    '''Step 3: Detrend the data'''
    history = lagged_returns(history, pipeline_args.args['data_lag'])

    '''Any sort of denoising, such is wavelet or fourier can go here'''

    '''Step 4: Split data into training/testing'''

    x_train, x_validation, x_test = train_test_split_custom(history, pipeline_args.args['train_size'],
                                                            pipeline_args.args['test_size'])

    '''Step 5: SS Transform'''
    if pipeline_args.args['mode'] == 'training' or pipeline_args.args['mode'] == 'continue':
        x_train, x_validation, x_test, y_train, y_validation, y_test = x_y_split(x_train, x_validation, x_test)
    elif pipeline_args.args['mode'] == 'prediction':
        x_train, x_validation, x_test, y_train, y_validation, y_test = x_y_split(x_train, x_validation,
                                                                                 history)  # This is a somewhat dirty workaround, this way during the predictions test always gets full data
    '''Step 6: Split data into x and y'''

    x_train, x_validation, x_test, _, _, _, SS_scaler = SS_transform(x_train, x_validation,
                                                                     x_test, y_train,
                                                                     y_validation, y_test,
                                                                     pipeline_args.args[
                                                                         'mode'], pipeline_args.args['interval'],
                                                                     pipeline_args.args['ticker'],
                                                                     SS_path=os.getenv('ss_path'))
    '''Step 7: PCA'''
    if pipeline_args.args['pca'] == True:
        x_train, x_validation, x_test = pca_reduction(x_train, x_validation, x_test)

    '''Step 8: Min-max scaler (-1 to 1 for sigmoid)'''
    x_train, x_validation, x_test, _, _, _, mm_scaler_y = min_max_transform(x_train, x_validation,
                                                                            x_test, y_train,
                                                                            y_validation, y_test,
                                                                            pipeline_args.args[
                                                                                'mode'], pipeline_args.args['interval'],
                                                                            pipeline_args.args['ticker'],
                                                                            os.getenv('mm_path'))
    '''Step 9: Create time-series data'''

    size = len(x_train)
    if pipeline_args.args['mode'] == 'training' or pipeline_args.args['mode'] == 'continue':  # This is to prevent errors during predictions due to timesteps
        x_train, y_train = build_timeseries(x_train, y_train, pipeline_args.args['time_steps'],
                                            pipeline_args.args['batch_size'],
                                            expand_dims=pipeline_args.args['expand_dims']
                                           )

        x_validation, y_validation = build_timeseries(x_validation, y_validation, pipeline_args.args['time_steps'],
                                                      pipeline_args.args['batch_size'],
                                                      expand_dims=pipeline_args.args['expand_dims']
                                                      )
    x_test, y_test = build_timeseries(x_test, y_test, pipeline_args.args['time_steps'],
                                      pipeline_args.args['batch_size'], expand_dims=pipeline_args.args['expand_dims']
                                      )

    return x_train, y_train, x_validation, y_validation, x_test, y_test, size
