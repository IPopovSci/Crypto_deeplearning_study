from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler,RobustScaler
import pandas as pd
from Arguments import args
import joblib
import numpy as np
import os

'''Standard scaler transform - it will substract the mean to center the data
as well as bring the standard deviation to 1. Will transform incoming pandas data into numpy in the process
The target columns are the first 5
returns transformed train,validation,test sets as well as the scaler'''
def SS_transform(x_train,x_validation,x_test):

    sc = StandardScaler()

    x_train_ss = sc.fit_transform(x_train)

    x_validation_ss = sc.transform(x_validation)

    x_test_ss = sc.transform(x_test)

    return x_train_ss,x_validation_ss,x_test_ss,sc

def min_max_sc_old(ticker,model='default'): #old version doesn't do robust scaling, use when predding from older models

    df_train = pd.read_csv(f"data/05_pca/{ticker}_train.csv")
    df_test = pd.read_csv(f"data/05_pca/{ticker}_test.csv")

    train_cols = df_train.keys()
    train_cols = list(train_cols)
    print(train_cols)
    train_cols.remove('Date')
    train_cols.remove('Close')

    #Have to apply separate scaler to the price, so we will separate
    x_close_train = df_train.loc[:,'Close'].values
    x_close_test = df_test.loc[:,'Close'].values

    del df_train['Close']
    del df_test['Close']



    x = df_train.loc[:, train_cols].values

    min_max_scaler = MinMaxScaler(feature_range=(0,0.8))

    #x_train = RobustScaler().fit_transform(x)
    x_train = min_max_scaler.fit_transform(x)
    x_test = min_max_scaler.transform(df_test.loc[:, train_cols])

    x_close_train = x_close_train.reshape(-1,1)
    x_close_test = x_close_test.reshape(-1, 1)

    min_max_scaler = MinMaxScaler(feature_range=(0, 0.8))
    x_close_train = min_max_scaler.fit_transform((x_close_train))
    x_close_test = min_max_scaler.transform(x_close_test)

    train_cols.insert(len(train_cols),'Close')

    x_train = np.concatenate([x_train,x_close_train],axis = 1)
    x_test = np.concatenate([x_test, x_close_test], axis=1)

    x_train_pd = pd.DataFrame(x_train,columns = train_cols)
    x_test_pd = pd.DataFrame(x_test,columns = train_cols)

    x_train_pd = x_train_pd.set_index(args['train_index'])
    x_test_pd = x_test_pd.set_index(args['test_index'])

    x_train_pd.to_csv(f"data/06_minmax/{ticker}_train.csv")
    x_test_pd.to_csv(f"data/06_minmax/{ticker}_test.csv")

    '''Saving the scaler - rework into a separate function?'''
    parent = 'C:/Users/Ivan/PycharmProjects/MlFinancialAnal/data/scalers/'
    directory = model
    path = os.path.join(parent,directory)
    try:
        os.mkdir(path)
    except:
        print(f'Folder {model} at {path} already exists')
    joblib.dump(min_max_scaler, f'data/scalers/{model}/{ticker}_mm.bin', compress=True)

    return x_train,x_test