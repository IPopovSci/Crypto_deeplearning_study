import pandas as pd
from sklearn.decomposition import PCA
from Arguments import args


def pca_reduction(ticker):
    n_components = args['n_components']
    train_cols = args['train_cols']

    df_train = pd.read_csv(f"data/04_SC/{ticker}_train.csv")
    df_test = pd.read_csv(f"data/04_SC/{ticker}_test.csv")

    df_no_close_train = df_train.drop(labels=['Close','Date'], axis=1)
    df_no_close_test = df_test.drop(labels=['Close', 'Date'], axis=1)

    train_cols = list(train_cols)
    train_cols.remove('Close')

    x_train = df_no_close_train.loc[:,train_cols].values
    x_test = df_no_close_test.loc[:, train_cols].values

    pca = PCA(n_components=n_components)

    pca_reduce_train = pca.fit_transform(x_train)
    pca_reduce_test = pca.transform(x_test)

    pcaDF_train = pd.DataFrame(data = pca_reduce_train)
    pcaDF_test = pd.DataFrame(data = pca_reduce_test)

    finalDF_train = pd.concat([pcaDF_train, df_train[['Close','Date']]], axis=1)
    finalDF_test = pd.concat([pcaDF_test, df_test[['Close','Date']]], axis=1)

    finalDF_train = finalDF_train.set_index('Date')
    finalDF_test = finalDF_test.set_index('Date')

    finalDF_train.to_csv(f"data/05_pca/{ticker}_train.csv")
    finalDF_test.to_csv(f"data/05_pca/{ticker}_test.csv")

