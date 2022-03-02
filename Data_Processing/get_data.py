import pandas as pd
import yfinance as yf
import os
from Arguments import args
# from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
from utility import join_files
from test_and_debug.test_arrays import dummy_timeseries
import cryptowatch as cw
from coinapi_rest_v1.restapi import CoinAPIv1

'''This module is for grabbing stock information from Yahoo Finance or other sources
Ticker_data grabs specific ticker, vix_data will grab only vix data and rename its columns so its easier to differentiate down the line'''


def ticker_data(ticker, start_date):
    col = ['Open', 'High', 'Low', 'Close', 'Volume']  # Relevant Columns
    args['target_features'] = col  # Send the features to predict name columns to the dictionary

    stock_object = yf.Ticker(ticker)  # yf stock object

    hist = stock_object.history(start=start_date)  # grabs history of stock

    hist_price = hist[col]  # Grabs only the relevant columns

    hist_price.to_csv(f'data/01_seed/{ticker}.csv')  # saves to csv

    return hist_price


'''Function for getting auxillilary data, such as VIX, bond yields, currency conversion values and so on
Works similarly to the above, and does automatic column renaming so as not to interfece with target_features columns,
accepts a list of tickers'''


def aux_data(df_main, aux_ticker_list, start_date):
    aux_total_df = df_main

    for ticker in aux_ticker_list:  # Loop over all the required tickers

        target_col = ['Open', 'High', 'Low', 'Close']  # Required columns to grab from yahoo object

        aux_data_object = yf.Ticker(f'{ticker}')  # Grab the yahoo ticker object

        aux_hist = aux_data_object.history(start=start_date)  # Grab history of the object with the indicated start date

        aux_hist = aux_hist[target_col]  # Grab target columns (Ignoring dividends and split columns)
        aux_hist.rename(columns={'Open': f'{ticker} Open', 'High': f'{ticker} High', 'Low': f'{ticker} Low',
                                'Close': f'{ticker} Close'}, inplace=True)

        aux_total_df = pd.concat([aux_total_df, aux_hist], axis=1)

    return aux_total_df


'''This function is from loading in prepared CSV data'''


def scv_data(path,filename):
    col = ['time', 'Open', 'High', 'Low', 'Close', 'Volume']
    #df = pd.read_csv(f'C:\\Users\\Ivan\\PycharmProjects\\MlFinancialAnal\\data\datasets\\{pair}\\{pair}.csv')
    df = pd.read_csv(f'{path}\\{filename}.csv')
    df = df[col]
    try:
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'},
                inplace=True)
    except:
        print('No columns to rename')
#    df['time'] = pd.to_datetime(df['time'], unit='ms')  # Unix to datetime conversion

    df.set_index('time', inplace=True)
    #print(df.head())

    return df


# '''Using Coingecko API to get minute data for the last day'''
# def coingecko_data(id,vscurrency,days):
#     cg = CoinGeckoAPI()
#     hist = cg.get_coin_market_chart_range_by_id(id=id,vs_currency=vscurrency,days=days)
#     # coins = cg.get_coins_list()
#     print(hist)
#     return hist
# coingecko_data('binancecoin','usd',1)

'''Get data from cryptowatch API'''
'''private key: x4p1k7VvUiRdd+5JLmE5SOm3P1cM/ZQyjPTE61lp
periods can be: 1m,5m,4h,1d and other (see api docs)
One thousand data points only'''



def cryptowatch_data(path,pair, periods,filename):
    cw.api_key = 'LZKL7ULRG322Z0793KU3'

    hist = cw.markets.get(f"BINANCE:{pair}", ohlc=True, periods=[f'{periods}'])

    hist_list = getattr(hist, f'of_{periods}')  # Calling a method on a class to get the desired interval
    #print(hist_list)
    col = ['time', 'Open', 'High', 'Low', 'Close', 'volume_a',
           'Volume']  # Volume is the volume in USDT in this case, volume_a is the volume in first currency (Currently using volume_a)
    df = pd.DataFrame(hist_list, columns=col)
    df.drop(['Volume'], axis=1, inplace=True)  # getting rid of first currency volume

    df.rename(columns={'volume_a': 'Volume'},
              inplace=True)

    df['time'] = pd.to_datetime(df['time'], unit='s').dt.strftime('%Y-%m-%dT%H:%M:%SZ')  # Unix to datetime conversion
    df.set_index('time', inplace=True)
    df.to_csv(f'{path}\\{filename}_cryptowatch.csv')

    return df

#cryptowatch_data('bnbusdt','5m')

'''Coinapi history grab - only 100 requests/day but 10000 points
Use for transfer learning step 1, then apply real world with cryptowatch api
useless for bnb-usd, has no data on it (It says it does but returns empty array)'''


import datetime, sys



def coinapi_data(path,filename,mode):
    api = CoinAPIv1('7E9EEDF3-DDEA-4176-8046-7BD4BFFE1670') #7E9EEDF3-DDEA-4176-8046-7BD4BFFE1670 #86B0BD38-FA87-4FA1-A4F1-A93819B09DF9 #EAB93C31-D483-44EC-BF6B-A095635C96EF #59821645-ABBF-43EF-884A-D613F3542507
    starting_date = datetime.date(2022,1, 16).isoformat() #Need to dynamically get 500000 minutes ago, this will do for now
#5C58BD30-97D0-4F9A-BB6C-A2AC22F63E86 #8BF8FA2A-C4B9-4651-8FE3-48B75B8CEE87
    #symbols = api.metadata_list_symbols({'filter_symbol_id':'binance_spot_bnb'}) # FCOIN_SPOT_BNB_USDT
    if mode == 'historical':
        ohlcv_historical = api.ohlcv_historical_data('BINANCE_SPOT_BNB_USDT', {'period_id': '1MIN', 'time_start': starting_date,'limit':'100000'})
    if mode == 'book':
        ohlcv_historical = api.orderbooks_historical_data('BINANCE_SPOT_BNB_USDT',
                                                     {'period_id': '5MIN', 'time_start': starting_date,
                                                      'limit': '100000'})
    else:
        ohlcv_historical = api.ohlcv_latest_data('BINANCE_SPOT_BNB_USDT', {'period_id': '1min','limit':'100000'})

    col = ['time_period_end','price_open','price_high','price_low','price_close', 'volume_traded']

    df = pd.DataFrame(ohlcv_historical, columns=col)

    df.rename(columns={'time_period_end':'time','price_open': 'Open', 'price_high': 'High', 'price_low': 'Low', 'price_close': 'Close', 'volume_traded': 'Volume'},
              inplace=True)

    df.set_index('time', inplace=True)

    df.to_csv(f'{path}\\{filename}_coinapi_book.csv')  # saves to csv

#coinapi_data('F:\MM\production\pancake_predictions\data','bnb_5m_pancake',mode='book')

def pancake_data(path,filename,big_update=False):
    if big_update == True:
        try:
            coinapi_data(path,filename,False)
        except:
            print('coinapi API key is out of requests')
    cryptowatch_data(path + '\load','bnbusdt', '5m',filename)
    df = join_files(path_load=path +'\load',path_save = path + '\save')
    return df

def testing_data(n):
    x = dummy_timeseries(n)

    x = x.T

    print(x.shape)

    x = pd.DataFrame(x)
    return x

