import pandas as pd
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")

'''This tests the stationarity of the data - whenever the probability distribution 
changes with progression of time or not. P values of near 0 indicate stationary series'''


def test_stationarity(timeseries):
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


'''Creates lagged returns to be used as both features and targets.
Targets are shifted back to avoid data-leaks.
Inputs: dataset (Pandas), ta (True/False)
Outputs: Pandas Dataframe'''


def lagged_returns(df, lags):
    for lag in lags:
        df[f'return_{lag}h'] = df['close'].pct_change(lag, axis=0)

    for t in lags:
        df[f'target_{t}h'] = df[f'return_{t}h'].shift(-t)
        df = df[[f'target_{t}h'] + [col for col in df.columns if
                                    col != f'target_{t}h']]  # Puts the return columns up front, for easier grabbing later

    return df
