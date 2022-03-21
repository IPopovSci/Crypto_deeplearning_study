import pandas as pd
from statsmodels.tsa.stattools import adfuller

# settings
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


'''Creates a new dataset filled with the differences of values of each day
This will make the data stationary (Use the above test to check), as well
as easier to work with for neural Networks
Inputs: dataset (Pandas), ta (True/False)
Outputs: Pandas Dataframe'''
'''Medium article discussion: Do covariants need to be stationary? Do we take the percent changes of ta features as well?'''

def lagged_returns(df,lags):
    pd.set_option('display.max_columns', None)

    lags = lags


    #cols = ['open','high','low','close','volume']

    #print(df.head(n=30))
    for lag in lags:

        df[f'return_{lag}h'] = df['close'].pct_change(lag,axis=0)

    #print(df.head(n=30))

    for t in lags:
        df[f'target_{t}h'] = df[f'return_{t}h'].shift(-t)
        df = df[[f'target_{t}h'] + [col for col in df.columns if col != f'target_{t}h']] #Puts the return columns up front, for easier grabbing later

    #print(df.head(n=30))


    #df_diff = df.iloc[1:, :]  # this drops the first row (For avoiding N/A)
    '''Debug options'''
    # pd.set_option('max_columns', None)

    # print(df_diff.head(n=30))

    return df


# invert differenced forecast #Second element (Value) is diff[i], first one is data[i]
def inverse_difference(last_ob, value):
    return value + last_ob
