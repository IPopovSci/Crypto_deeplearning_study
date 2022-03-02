# Basic packages
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import Series
from statsmodels.tsa.stattools import adfuller, acf, pacf, arma_order_select_ic
import datetime as dt

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
as easier to work with for neural networks'''


def row_difference(df,ta):
    df_data = df.iloc[:,:5]





    df_ta = df.iloc[:,5:]

    #df_ta = np.log1p(df_ta) #take log of ta? Does that makes sense? To avoid any stat imbalance
    '''Implementing categorization on data'''
    #df_data_lg = np.log1p(df_data)  # take a log, to smooth out any shmuckery

    #df_ta = np.log1p(df_ta)


    df_diff_data = df_data.pct_change(axis=0)

    print(df_diff_data[-30:])

    #df_diff_ta = df_ta.pct_change(axis=0)
    #df_diff_ta = df_diff_ta.fillna(0)
    if ta == True:
        df_diff = pd.concat([df_diff_data,df_ta],axis=1)
    else:
        df_diff = pd.concat([df_diff_data], axis=1)



    df_diff = df_diff.iloc[1:, :] #this drops the first row (For avoiding N/A)
    '''Debug options'''
    #pd.set_option('max_columns', None)

    #print(df_diff.head(n=30))

    return df_diff


# invert differenced forecast #Second element (Value) is diff[i], first one is data[i]
def inverse_difference(last_ob, value):
    return value + last_ob
