import pandas as pd
import yfinance as yf
from datetime import datetime
import cryptowatch as cw
from coinapi_rest_v1.restapi import CoinAPIv1
import datetime
from pipeline.pipelineargs import PipelineArgs
from Networks.network_config import NetworkParams

pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()

'''This module is for grabbing stock information from Yahoo Finance or other sources
Ticker_data grabs specific ticker, vix_data will grab only vix data and rename its columns so its easier to differentiate down the line'''


def ticker_data(ticker, start_date):
    col = ['Open', 'High', 'Low', 'Close', 'Volume']  # Relevant Columns
    # pipeline_args.args['target_features'] = col  # Send the features to predict name columns to the dictionary

    stock_object = yf.Ticker(ticker)  # yf stock object

    hist = stock_object.history(start=start_date)  # grabs history of stock

    hist_price = hist[col]  # Grabs only the relevant columns

    hist_price.to_csv(f'data/01_seed/{ticker}.csv')  # saves to csv

    return hist_price


'''Function for getting auxillilary data, such as VIX, bond yields, currency conversion values and so on
Works similarly to the above, and does automatic column renaming so as not to interfere with target_features columns,
accepts a list of tickers
This data is daily, so will only work with daily, or longer time intervals (Unless sparse array is OK)'''


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


def scv_data(ticker, path, interval):
    col = ['time', 'open', 'high', 'low', 'close', 'volume']
    # df = pd.read_csv(f'C:\\Users\\Ivan\\PycharmProjects\\MlFinancialAnal\\data\datasets\\{pair}\\{pair}.csv')
    df = pd.read_csv(f'{path}/{interval}/{ticker}.csv')
    df = df[col]

    try:
        df['time'] = pd.to_datetime(df['time'], unit='ms')  # Unix to datetime conversion
    except:  # todo: catch specific exception
        print('No need to convert to datetime')

    df.set_index('time', inplace=True)

    return df


'''Get data from cryptowatch API
periods can be: 1m,5m,4h,1d and other (see api docs)
One thousand data points only'''


def cryptowatch_data(pair, periods):
    cw.api_key = pipeline_args.args['cryptowatch_key']

    # Sometimes data from Kaggle has different pair differentiator, need to implement custom swaps.
    if pair == 'btcusd':
        pair = 'btcusdt'
    elif pair == 'ethusd':
        pair = 'ethusdt'

    hist = cw.markets.get(f"BINANCE:{pair}", ohlc=True, periods=[f'{periods}'])

    hist_list = getattr(hist, f'of_{periods}')  # Calling a method on a class to get the desired interval

    col = ['time', 'Open', 'High', 'Low', 'Close', 'volume_a',
           'Volume']  # Volume is the volume in USDT in this case, volume_a is the volume in first currency (Currently using volume_a)
    df = pd.DataFrame(hist_list, columns=col)
    df.drop(['Volume'], axis=1, inplace=True)  # getting rid of first currency volume

    df.rename(columns={'volume_a': 'volume', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'},
              inplace=True)

    df['time'] = pd.to_datetime(df['time'], unit='s').dt.strftime('%Y-%m-%dT%H:%M:%SZ')  # Unix to datetime conversion
    df.set_index('time', inplace=True)

    return df




'''Coinapi history grab - only 100 requests/day but 10000 points
Use for transfer learning step 1, then apply real world with cryptowatch api
useless for bnb-usd, has no data on it (It says it does but returns empty array)
This function is not finished, nor is it used, if you want to use it modify it'''


def coinapi_data(mode):
    api = CoinAPIv1('your_key_here')
    starting_date = datetime.date(2022, 1, 16).isoformat()

    if mode == 'historical':
        ohlcv_historical = api.ohlcv_historical_data('BINANCE_SPOT_BNB_USDT',
                                                     {'period_id': '1MIN', 'time_start': starting_date,
                                                      'limit': '100000'})
    elif mode == 'book':
        ohlcv_historical = api.orderbooks_historical_data('BINANCE_SPOT_BNB_USDT',
                                                          {'period_id': '5MIN', 'time_start': starting_date,
                                                           'limit': '100000'})
    else:
        ohlcv_historical = api.ohlcv_latest_data('BINANCE_SPOT_BNB_USDT', {'period_id': '1min', 'limit': '100000'})

    col = ['time_period_end', 'price_open', 'price_high', 'price_low', 'price_close', 'volume_traded']

    df = pd.DataFrame(ohlcv_historical, columns=col)

    df.rename(columns={'time_period_end': 'time', 'price_open': 'open', 'price_high': 'high', 'price_low': 'low',
                       'price_close': 'close', 'volume_traded': 'volume'},
              inplace=True)

    df.set_index('time', inplace=True)
