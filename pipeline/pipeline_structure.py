from Data_Processing.get_data import scv_data, cryptowatch_data
from Data_Processing.ta_feature_add import add_ta
from Data_Processing.create_lags import lagged_returns
from Data_Processing.data_split import train_test_split_custom, x_y_split
from Data_Processing.data_scaling import SS_transform, min_max_transform
from Data_Processing.PCA import pca_reduction
from Data_Processing.build_timeseries import build_timeseries
import os
from dotenv import load_dotenv
from utility import structure_create

load_dotenv()


'''This is the pipeline function, which will call upon required functions to load and process the data'''


def pipeline(pipeline_args):
    '''Step 0: Create folder structure, if needed'''
    structure_create()
    '''Step 1: Get Data'''
    if pipeline_args.args['mode'] == 'training':
        history = scv_data(pipeline_args.args['ticker'], os.getenv('data_path'), pipeline_args.args['interval'])
    elif pipeline_args.args['mode'] == 'prediction':
        history = cryptowatch_data(pipeline_args.args['ticker'], pipeline_args.args['interval'])

    '''Any additional feature extractors can go here'''

    '''Step 2: Apply TA Analysis'''
    if pipeline_args.args['ta'] == True:
        history = add_ta(history)  # The columns names can be acessed from network_config 'train_cols'

    '''Step 3: Detrend the data'''
    history = lagged_returns(history,pipeline_args.args['data_lag'])

    '''Any sort of denoising, such is wavelet or fourier can go here'''

    '''Step 4: Split data into training/testing'''

    x_train, x_validation, x_test = train_test_split_custom(history,pipeline_args.args['train_size'],pipeline_args.args['test_size'])



    '''Step 5: SS Transform'''
    if pipeline_args.args['mode'] == 'training':
        x_train, x_validation, x_test, y_train, y_validation, y_test = x_y_split(x_train, x_validation, x_test)
    elif pipeline_args.args['mode'] == 'prediction':
        x_train, x_validation, x_test, y_train, y_validation, y_test = x_y_split(x_train, x_validation, history) #This is a somewhat dirty workaround, this way during the predictions test always gets full data

    '''Step 6: Split data into x and y'''

    x_train, x_validation, x_test, _, _, _, SS_scaler = SS_transform(x_train, x_validation,
                                                                                           x_test, y_train,
                                                                                           y_validation, y_test,
                                                                                           pipeline_args.args[
                                                                                               'mode'],pipeline_args.args['interval'],pipeline_args.args['ticker'],
                                                                                           SS_path=os.getenv('ss_path'))
    '''Step 7: PCA'''
    if pipeline_args.args['pca'] == True:
        x_train, x_validation, x_test = pca_reduction(x_train, x_validation, x_test)

    '''Step 8: Min-max scaler (-1 to 1 for sigmoid)'''
    x_train, x_validation, x_test, _, _, _, mm_scaler_y = min_max_transform(x_train, x_validation,
                                                                                                  x_test, y_train,
                                                                                                  y_validation, y_test,
                                                                                                  pipeline_args.args[
                                                                                                      'mode'],pipeline_args.args['interval'],pipeline_args.args['ticker'],
                                                                                                  os.getenv('mm_path'))
    '''Step 9: Create time-series data'''

    size = len(x_train) - 1
    if pipeline_args.args['mode'] == 'training': #This is to prevent errors during predictions due to timesteps
        x_train, y_train = build_timeseries(x_train, y_train,pipeline_args.args['time_steps'],pipeline_args.args['batch_size'],expand_dims=pipeline_args.args['expand_dims'],data_lag = pipeline_args.args['data_lag'])

        x_validation, y_validation = build_timeseries(x_validation, y_validation,pipeline_args.args['time_steps'],pipeline_args.args['batch_size'],expand_dims=pipeline_args.args['expand_dims'],data_lag = pipeline_args.args['data_lag'])
    x_test, y_test = build_timeseries(x_test, y_test,pipeline_args.args['time_steps'],pipeline_args.args['batch_size'],expand_dims=pipeline_args.args['expand_dims'],data_lag = pipeline_args.args['data_lag'])

    return x_train, y_train, x_validation, y_validation, x_test, y_test, size
