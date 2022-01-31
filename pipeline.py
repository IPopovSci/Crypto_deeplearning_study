from Arguments import args
from Data_Processing.get_data import ticker_data, aux_data, scv_data, cryptowatch_data,coinapi_data
from Data_Processing.ta_feature_add import add_ta
from Data_Processing.Detrending import row_difference
from Data_Processing.data_split import train_test_split_custom, x_y_split,x_y_split_small
from Data_Processing.data_scaling import SS_transform,min_max_transform,SS_transform_small,min_max_transform_small
from Data_Processing.PCA import pca_reduction,pca_reduction_small
from Data_Processing.build_timeseries import build_timeseries
import numpy as np
import sys

ticker = args['ticker']
BATCH_SIZE = args['batch_size']
start_date = args['starting_date']
np.set_printoptions(threshold=sys.maxsize)

'''This is the pipeline function, which will call upon required functions to load and process the data'''
def data_prep(data_from,initial_training=True,batch=True,SS_path = [],MM_path = []):
    '''Step 1: Get Data'''
    if data_from == 'Yahoo':
        history = ticker_data(ticker, start_date)
        history = aux_data(history, ['CL=F', 'GC=F', '^VIX', '^TNX'], start_date)  # Get any extra data
    elif data_from == 'CSV':
        history = scv_data(ticker)
        #history = history[:150000] #This is bad, but the dataset is too big, I'm too newb, and the beggining of the dataset has a lot of gaps anyways
    elif data_from == 'cryptowatch':
        history = cryptowatch_data(ticker,'5m')
    print('Got the Data!')
    '''Step 2: Apply TA Analysis'''
    ta_data = add_ta(history, ticker)  # The columns names can be acessed from arguments 'train_cols'
    print('ta = applied')
    '''Step 3: Detrend the data'''
    one_day_detrend = row_difference(ta_data)
    print('detrending = donzo')
    '''Step 4: Split data into training/testing'''
    x_train, x_validation, x_test = train_test_split_custom(one_day_detrend)

    print('AND I SPLIT IT IN HALF')
    '''Step 5: SS Transform'''
    x_train, x_validation, x_test, y_train, y_validation, y_test = x_y_split(x_train, x_validation, x_test)
    print('test after inversing:', y_test[-25:, :])
    print('SS, but this aint 1942')
    '''Step 6: Split data into x and y'''

    #x_train, x_validation, x_test,y_train,y_validation,y_test, SS_scaler = SS_transform(x_train, x_validation, x_test, y_train,y_validation,y_test, initial_training, SS_path)
    print('I SPLIT IT IN HALF, AGAIN!')
    '''Step 7: PCA'''
    #x_train, x_validation, x_test = pca_reduction(x_train, x_validation, x_test)
    print('PCA done')
    '''Step 8: Min-max scaler (-1 to 1 for sigmoid)'''
    #x_train, x_validation, x_test, y_train, y_validation, y_test, mm_scaler_y = min_max_transform(x_train, x_validation,
                                                                                                  # x_test, y_train,
                                                                                                  # y_validation, y_test,initial_training,MM_path)

    print('Min-maxed to the tits')
    '''Step 9: Create time-series data'''
    size = len(x_train) - 1
    if batch == False:
        x_train, y_train = build_timeseries(x_train, y_train)

        x_validation, y_validation = build_timeseries(x_validation, y_validation)
        x_test, y_test = build_timeseries(x_test, y_test)
    else:
        x_validation, y_validation = build_timeseries(x_validation, y_validation)
        x_test, y_test = build_timeseries(x_test, y_test)
    print('timeseries = built')
    return x_train, y_train, x_validation, y_validation, x_test, y_test, size
'''separate data prep, without timeseries building, for large datasets'''
# def data_prep_batch_1(data_from):
#     '''Step 1: Get Data'''
#     if data_from == 'Yahoo':
#         history = ticker_data(ticker, start_date)
#         history = aux_data(history, ['CL=F', 'GC=F', '^VIX', '^TNX'], start_date)  # Get any extra data
#     elif data_from == 'CSV':
#         history = scv_data(ticker)
#         history = history[100000:] #This is bad, but the dataset is too big, I'm too newb, and the beggining of the dataset has a lot of gaps anyways
#     elif data_from == 'cryptowatch':
#         history = cryptowatch_data(ticker,'5m')
#     print('Got the Data!')
#     '''Step 2: Apply TA Analysis'''
#     ta_data = add_ta(history, ticker)  # The columns names can be acessed from arguments 'train_cols'
#     print('ta = applied')
#     '''Step 3: Detrend the data'''
#     one_day_detrend = row_difference(ta_data)
#     print('detrending = donzo')
#     '''Step 4: Split data into training/testing'''
#     x_train, x_validation, x_test = train_test_split_custom(one_day_detrend)
#     print('AND I SPLIT IT IN HALF')
#     '''Step 5: SS Transform'''
#
#     x_train, x_validation, x_test, SS_scaler = SS_transform(x_train, x_validation, x_test)
#
#     print('SS, but this aint 1942')
#     '''Step 6: Split data into x and y'''
#     x_train, x_validation, x_test, y_train, y_validation, y_test = x_y_split(x_train, x_validation, x_test)
#     print('I SPLIT IT IN HALF, AGAIN!')
#     '''Step 7: PCA'''
#
#     x_train, x_validation, x_test = pca_reduction(x_train, x_validation, x_test)
#     print('PCA done')
#     '''Step 8: Min-max scaler (-1 to 1 for sigmoid)'''
#     # x_train, x_validation, x_test, y_train, y_validation, y_test, mm_scaler_y = min_max_transform(x_train, x_validation,
#     #                                                                                               x_test, y_train,
#     #                                                                                               y_validation, y_test)
#     #print(x_train)
#     print('Min-maxed to the tits')
#     size = len(x_train) - 1
#     #print(mm_scaler_y.inverse_transform(x_train))
#     '''Step 9: Create time-series data'''
#
#     x_validation, y_validation = build_timeseries(x_validation, y_validation) #Can make validation and set arbitrary small, so just the x_train, y_train that need to be cut
#     x_test, y_test = build_timeseries(x_test, y_test)
#     print('timeseries = built')
#     return x_train, y_train, x_validation, y_validation, x_test, y_test, size
'''Isolated build timeseries function to load large datasets'''
def data_prep_batch_2(x_train,y_train,i,j):
    x_train_save, y_train_save = build_timeseries(x_train[i:j], y_train[i:j])
    print(f'Loaded {ticker} {i} to {j}')
    return x_train_save,y_train_save
''''''
def data_prep_transfer(data_from):
    '''Step 1: Get Data'''
    if data_from == 'Yahoo':
        history = ticker_data(ticker, start_date)
        history = aux_data(history, ['CL=F', 'GC=F', '^VIX', '^TNX'], start_date)  # Get any extra data
    elif data_from == 'CSV':
        history = scv_data(ticker)
        #history = history[450000:] #This is bad, but the dataset is too big, I'm too newb, and the beggining of the dataset has a lot of gaps anyways
    elif data_from == 'cryptowatch':
        history = cryptowatch_data(ticker, '5m')
    elif data_from == 'coinapi':
        history = coinapi_data(historical=False)
    print('Got the Data!')
    '''Step 2: Apply TA Analysis'''
    ta_data = add_ta(history, ticker)  # The columns names can be acessed from arguments 'train_cols'
    print('ta = applied')
    '''Step 3: Detrend the data'''
    one_day_detrend = row_difference(ta_data)
    print('detrending = donzo')
    '''Step 4: Split data into training/testing'''
    #x_train, x_validation, x_test = train_test_split_custom(one_day_detrend)
    print('AND I SPLIT IT IN HALF')
    '''Step 5: SS Transform'''
    x_train, SS_scaler = SS_transform_small(one_day_detrend)
    print('SS, but this aint 1942')
    '''Step 6: Split data into x and y'''
    x_train, y_train = x_y_split_small(x_train)
    print('I SPLIT IT IN HALF, AGAIN!')
    '''Step 7: PCA'''
    x_train = pca_reduction_small(x_train)
    print('PCA done')
    '''Step 8: Min-max scaler (-1 to 1 for sigmoid)'''
    x_train, y_train, mm_scaler_y = min_max_transform_small(x_train,
                                                                                                  y_train)
    print('Min-maxed to the tits')
    '''Step 9: Create time-series data'''
    x_train, y_train = build_timeseries(x_train, y_train)

    print('timeseries = built')
    return x_train, y_train