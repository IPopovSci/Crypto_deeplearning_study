import ta
import pandas as pd
from Arguments import args

def add_ta(ticker_data,ticker):
    df = ticker_data  # Load Data In
    #print('TA - loaded data')
    df = ta.add_all_ta_features(df, open=f"Open", high=f"High", low=f"Low", close=f"Close", volume=f"Volume",
                                fillna=True,vectorized=True)  # Add all the ta!
    #print('TA - applied TA')
    #df = df.set_index('Date')

    args['train_cols'] = df.keys() #Column names of all the ta analysis + original columns
    #print('TA - got the keys')

    # df.to_csv(f"data/02_ta/{ticker}.csv")  # Save ta data along with original to csv

    return df