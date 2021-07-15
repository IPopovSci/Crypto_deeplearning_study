import pandas as pd
import yfinance as yf

def ticker_data(ticker):
    col = ['Open', 'High', 'Low', 'Close', 'Volume']  # Relevant Columns
    stock_object = yf.Ticker(ticker)  # yf stock object
    vix = yf.Ticker('^VIX')

    hist = stock_object.history(start="1982-01-02")  # grabs history of stock
    vix_hist = vix.history(start="2000-12-02")


    vix_hist = vix_hist[col]
    hist_price = hist[col]

    #hist_price = pd.concat([hist_price, vix_hist],axis=1)

    hist_price.to_csv(f'data/01_seed/{ticker}.csv') #saves to csv