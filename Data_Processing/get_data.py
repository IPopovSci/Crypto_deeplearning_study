import pandas as pd
import yfinance as yf
import os
from Arguments import args
# from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta

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


def scv_data(pair):
    col = ['time', 'open', 'high', 'low', 'close', 'volume']
    df = pd.read_csv(f'C:\\Users\\Ivan\\PycharmProjects\\MlFinancialAnal\\data\datasets\{pair}\{pair}.csv')
    df = df[col]
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'},
              inplace=True)  # Data from kaggle doesn't have capitalization, this is a fix
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
import cryptowatch as cw


def cryptowatch_data(pair, periods):
    cw.api_key = 'LZKL7ULRG322Z0793KU3'

    hist = cw.markets.get(f"BINANCE:{pair}", ohlc=True, periods=[f'{periods}'])

    hist_list = getattr(hist, f'of_{periods}')  # Calling a method on a class to get the desired interval

    col = ['time', 'Open', 'High', 'Low', 'Close', 'volume_a',
           'Volume']  # Volume is the volume in USDT in this case, volume_a is the volume in first currency
    df = pd.DataFrame(hist_list, columns=col)
    df.drop(['volume_a'], axis=1, inplace=True)  # getting rid of first currency volume

    df['time'] = pd.to_datetime(df['time'], unit='s')  # Unix to datetime conversion
    df.set_index('time', inplace=True)

    print(df)
    return df

'''Coinapi history grab - only 100 requests/day but 10000 points
Use for transfer learning step 1, then apply real world with cryptowatch api
useless for bnb-usd, has no data on it (It says it does but returns empty array)'''

from coinapi_rest_v1.restapi import CoinAPIv1
import datetime, sys



def coinapi_data(historical):
    api = CoinAPIv1('8BF8FA2A-C4B9-4651-8FE3-48B75B8CEE87') #7E9EEDF3-DDEA-4176-8046-7BD4BFFE1670 #86B0BD38-FA87-4FA1-A4F1-A93819B09DF9 #EAB93C31-D483-44EC-BF6B-A095635C96EF #59821645-ABBF-43EF-884A-D613F3542507
    starting_date = datetime.date(2020, 1, 4).isoformat() #Need to dynamically get 500000 minutes ago, this will do for now
#5C58BD30-97D0-4F9A-BB6C-A2AC22F63E86 #8BF8FA2A-C4B9-4651-8FE3-48B75B8CEE87
    #symbols = api.metadata_list_symbols({'filter_symbol_id':'binance_spot_bnb'}) # FCOIN_SPOT_BNB_USDT
    if historical == True:
        ohlcv_historical = api.ohlcv_historical_data('BINANCE_SPOT_BNB_USDT', {'period_id': '1MIN', 'time_start': starting_date,'limit':'100000'})
    else:
        ohlcv_historical = api.ohlcv_latest_data('BINANCE_SPOT_BNB_USDT', {'period_id': '5min','limit':'100000'})

    col = ['time_period_start','price_open','price_high','price_low','price_close', 'volume_traded']
    df = pd.DataFrame(ohlcv_historical, columns=col)
    df.to_csv(f'F:\MM\Data\BNBUSDT\\bnbusdt_2020.csv')  # saves to csv
    print(ohlcv_historical)
#coinapi_data(historical=True)

#COINDCX_SPOT_BNB_LINK - FATBTC_SPOT_LINK_USDT,EXRATES_SPOT_LINK_USDT,FATBTC_SPOT_LINK_USDT
# COINDCX_SPOT_BNB_ROSE -COINDCX_SPOT_BUSD_ROSE
#COINDCX_SPOT_BNB_LUNA - COINDCX_SPOT_ETH_LUNA,COINDCX_SPOT_BUSD_LUNA
#COINDCX_SPOT_BNB_BETA - COINDCX_SPOT_USDT_BETA
#COINDCX_SPOT_BNB_KP3R -  COINDCX_SPOT_BUSD_KP3R
#COINDCX_SPOT_BNB_DAR - COINDCX_SPOT_USDT_DAR
#