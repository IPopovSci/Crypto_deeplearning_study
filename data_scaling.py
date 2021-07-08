from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from Arguments import args

def SS_transform(ticker):
    train_cols = args['train_cols']

    df_train = pd.read_csv(f"data/03_split/{ticker}_train.csv")
    df_test = pd.read_csv(f"data/03_split/{ticker}_test.csv")

    sc = StandardScaler()

    x = df_train.loc[:, train_cols].values
    x_train = sc.fit_transform(x)
    x_test = sc.transform(df_test.loc[:, train_cols])

    x_train_pd = pd.DataFrame(x_train,columns = train_cols)
    x_test_pd = pd.DataFrame(x_test,columns = train_cols)

    x_train_pd = x_train_pd.set_index(args['train_index'])
    x_test_pd = x_test_pd.set_index(args['test_index'])

    x_train_pd.to_csv(f"data/04_SC/{ticker}_train.csv")
    x_test_pd.to_csv(f"data/04_SC/{ticker}_test.csv")

def min_max_sc(ticker):

    df_train = pd.read_csv(f"data/05_pca/{ticker}_train.csv")
    df_test = pd.read_csv(f"data/05_pca/{ticker}_test.csv")

    train_cols = df_train.keys()
    train_cols = list(train_cols)
    train_cols.remove('Date')

    x = df_train.loc[:, train_cols].values
    min_max_scaler = MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x)
    x_test = min_max_scaler.transform(df_test.loc[:, train_cols])

    x_train_pd = pd.DataFrame(x_train,columns = train_cols)
    x_test_pd = pd.DataFrame(x_test,columns = train_cols)

    x_train_pd = x_train_pd.set_index(args['train_index'])
    x_test_pd = x_test_pd.set_index(args['test_index'])

    x_train_pd.to_csv(f"data/06_minmax/{ticker}_train.csv")
    x_test_pd.to_csv(f"data/06_minmax/{ticker}_test.csv")

    return x_train,x_test