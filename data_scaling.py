from sklearn.preprocessing import StandardScaler
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