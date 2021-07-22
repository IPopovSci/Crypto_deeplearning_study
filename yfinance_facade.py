import pandas as pd
import yfinance as yf


def ticker_data(ticker):
    col = ['Open', 'High', 'Low', 'Close', 'Volume']  # Relevant Columns
    vix_col = ['Close']

    stock_object = yf.Ticker(ticker)  # yf stock object
    vix = yf.Ticker('^VIX')

    hist = stock_object.history(start="1982-01-03")  # grabs history of stock
    vix_hist = vix.history(start=hist.index[0])

    vix_hist = vix_hist[vix_col]
    vix_hist['Vix Close'] = vix_hist['Close']
    del vix_hist['Close']

    hist_price = hist[col]


    hist_price = pd.concat([vix_hist,hist_price],axis=1)


    hist_price.to_csv(f'data/01_seed/{ticker}.csv')  # saves to csv
