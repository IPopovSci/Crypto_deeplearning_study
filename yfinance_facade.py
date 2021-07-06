import pandas as pd
import yfinance as yf

def ticker_data(ticker):
    col = ['Open', 'High', 'Low', 'Close', 'Volume']  # Relevant Columns
    stock_object = yf.Ticker(ticker)  # yf stock object

    hist = stock_object.history(start="1982-12-02")  # grabs history of stock

    hist_price = hist[col]
    hist_price.to_csv(f'data/01_seed/{ticker}.csv') #saves to csv