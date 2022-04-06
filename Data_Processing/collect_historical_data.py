import pandas as pd
import cryptowatch as cw
from datetime import datetime,date

now = date.today()

current_time = now.strftime("%H:%M:%S")

def cryptowatch_data_save_to_csv(pair, periods, save_to_csv=True):
    cw.api_key = 'LZKL7ULRG322Z0793KU3'

    hist = cw.markets.get(f"BINANCE:{pair}", ohlc=True, periods=[f'{periods}'])

    hist_list = getattr(hist, f'of_{periods}')  # Calling a method on a class to get the desired interval

    col = ['time', 'Open', 'High', 'Low', 'Close', 'volume_a',
           'Volume']  # Volume is the volume in USDT in this case, volume_a is the volume in first currency
    df = pd.DataFrame(hist_list, columns=col)
    df.drop(['volume_a'], axis=1, inplace=True)  # getting rid of first currency volume

    df['time'] = pd.to_datetime(df['time'], unit='s')  # Unix to datetime conversion
    df.set_index('time', inplace=True)
    if save_to_csv==True:
        df.to_csv(f'F:/MM/historical_data/bnbusdt/{now}.csv')

    return df

