from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from Arguments import args
import joblib
import numpy as np
import os

def save_scaler(sc, model='default'):
    '''Saving the scaler - rework into a separate function?'''
    parent = args['parent']
    directory = model
    path = os.path.join(parent,directory)
    try:
        os.mkdir(path)
    except:
        print(f'Folder {model} at {path} already exists')
    joblib.dump(sc, f'data/scalers/{model}/{ticker}_sc.bin', compress=True)

def SS_transform(ticker,model='default'):
    train_cols = list(args['train_cols'])

    df_train = pd.read_csv(f"data/03_split/{ticker}_train.csv")
    df_test = pd.read_csv(f"data/03_split/{ticker}_test.csv")

    x_close_train = df_train.loc[:,'Close'].values
    x_close_test = df_test.loc[:,'Close'].values

    del df_train['Close']
    del df_test['Close']
    train_cols.remove('Close')

    sc = StandardScaler()

    x = df_train.loc[:, train_cols].values

    x_train = sc.fit_transform(x)
    x_test = sc.transform(df_test.loc[:, train_cols])

    x_close_train = x_close_train.reshape(-1,1)
    x_close_test = x_close_test.reshape(-1, 1)

    x_close_train = sc.fit_transform(x_close_train)
    x_close_test = sc.transform(x_close_test)

    x_train = np.concatenate([x_train,x_close_train],axis = 1)
    x_test = np.concatenate([x_test, x_close_test], axis=1)

    train_cols.insert(len(train_cols), 'Close')

    x_train_pd = pd.DataFrame(x_train,columns = train_cols)
    x_test_pd = pd.DataFrame(x_test,columns = train_cols)

    x_train_pd = x_train_pd.set_index(args['train_index'])
    x_test_pd = x_test_pd.set_index(args['test_index'])

    x_train_pd.to_csv(f"data/04_SC/{ticker}_train.csv")
    x_test_pd.to_csv(f"data/04_SC/{ticker}_test.csv")

    save_scaler(sc, model)

def min_max_sc(ticker,model='default'):

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

    min_max_scaler = MinMaxScaler(feature_range=(0,0.2))
    x_train = min_max_scaler.fit_transform(x)
    x_test = min_max_scaler.transform(df_test.loc[:, train_cols])

    x_close_train = x_close_train.reshape(-1,1)
    x_close_test = x_close_test.reshape(-1, 1)

    min_max_scaler = MinMaxScaler(feature_range=(0, 0.2))
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

def load_sc(ticker, model='default',scaler='mm'):
    print("LOADING=> "f'data/scalers/{model}/{ticker}_{scaler}.bin')
    return joblib.load(f'data/scalers/{model}/{ticker}_{scaler}.bin')

def unscale_data(ticker,y_pred_lstm,y_test_t):
    mm = load_sc(ticker, model='default', scaler='mm')
    sc = load_sc(ticker, model='default', scaler='sc')

    y_pred_lstm = y_pred_lstm.reshape(-1, 1)

    y_pred_lstm_inv_mm = mm.inverse_transform(y_pred_lstm)
    y_pred_lstm_inv_sc = sc.inverse_transform(y_pred_lstm_inv_mm)

    y_test_t = y_test_t.reshape(-1, 1)

    y_test_t_inv_mm = mm.inverse_transform(y_test_t)
    y_test_t_inv_sc = sc.inverse_transform(y_test_t_inv_mm)

    return y_pred_lstm_inv_sc,y_test_t_inv_sc

def unscale_data_np(ticker,y_test_t): #preediction free version

    mm = load_sc(ticker, model='default', scaler='mm')
    sc = load_sc(ticker, model='default', scaler='sc')

    y_test_t = y_test_t.reshape(-1, 1)

    y_test_t_inv_mm = mm.inverse_transform(y_test_t)
    y_test_t_inv_sc = sc.inverse_transform(y_test_t_inv_mm)

    return y_test_t_inv_sc