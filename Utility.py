import pandas as pd
from Data_Processing.get_data import scv_data
from Arguments import args

def one_to_five(data):
    ohlc = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    df = data.resample('5min', base=0).apply(ohlc)
    df.dropna(inplace=True)

    print(df)

ticker = args['ticker']
history = scv_data(ticker)
#print(history.head())
one_to_five(history)