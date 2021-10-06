import pandas as pd
import yfinance as yf
from Arguments import args

'''This module is for grabbing stock information from Yahoo Finance
Ticker_data grabs specific ticker, vix_data will grab only vix data and rename its columns so its easier to differentiate down the line'''
start_date = args['starting_date']
def ticker_data(ticker,start_date):
    col = ['Open', 'High', 'Low', 'Close', 'Volume']  # Relevant Columns

    stock_object = yf.Ticker(ticker)  # yf stock object

    hist = stock_object.history(start=start_date)  # grabs history of stock

    hist_price = hist[col] #Grabs only the relevant columns

    hist_price.to_csv(f'data/01_seed/{ticker}.csv')  # saves to csv

    return hist_price


def vix_data(start_date):
    vix_col = ['Open', 'High', 'Low', 'Close']
    vix = yf.Ticker('^VIX')

    vix_hist = vix.history(start=start_date)

    vix_hist = vix_hist[vix_col]

    vix_hist.rename(columns = {'Open':'Vix Open','High':'Vix High','Low':'Vix Low','Close':'Vix Close'},inplace=True)

    return vix_hist

