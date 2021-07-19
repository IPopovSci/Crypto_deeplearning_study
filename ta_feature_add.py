import ta
import pandas as pd
from Arguments import args

def add_ta(ticker):
    df = pd.read_csv(f"data/01_seed/{ticker}.csv")  # Read invididual files
    df = ta.add_all_ta_features(df, open=f"Open", high=f"High", low=f"Low", close=f"Close", volume=f"Volume",
                                fillna=True)  # Add all the ta!
    df = df.set_index('Date')

    args['train_cols'] = df.keys() #This needs to be taken oujt separately

    df_close = df['Close']
    del df['Close']
    df.insert(len(args['train_cols'])-1, 'Close', df_close) #Moving the Close column to the end of the features (For scaling purposes later)

    args['train_cols'] = df.keys() #Get a new list since we moved closed to end

    df.to_csv(f"data/02_ta/{ticker}.csv")  # Save ta data along with original to csv