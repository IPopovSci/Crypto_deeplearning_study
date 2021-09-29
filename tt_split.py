from sklearn.model_selection import train_test_split
import pandas as pd
from Arguments import args

def train_test_split_custom(ticker):
    df = pd.read_csv(f"data/02_ta/{ticker}.csv")
    df_train, df_test = train_test_split(df, train_size=0.96, test_size=0.04, shuffle=False)

    df_train = df_train.set_index('Date')
    df_test = df_test.set_index('Date')

    args['train_index'] = df_train.index
    args['test_index'] = df_test.index


    df_train.to_csv(f"data/03_split/{ticker}_train.csv")
    df_test.to_csv(f"data/03_split/{ticker}_test.csv")


