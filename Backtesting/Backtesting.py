import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from scipy.stats import spearmanr,kendalltau
from pipeline.pipelineargs import PipelineArgs
from dotenv import load_dotenv
from Networks.network_config import NetworkParams
import pandas as pd

from pathlib import Path
from time import time
import datetime

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from utility import scale
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

import os
load_dotenv()
pipeline_args = PipelineArgs.get_instance()

'''Function that calculates the amount of equal signs between y_true and y_pred
Accepts: y_true,y_pred.
Returns: Print out of the information.'''

def correct_signs(y_true,y_pred):
    # if pipeline_args.args['expand_dims'] == False:
    #     y_pred = y_pred[:,-1,:]
    y_total = np.empty(5)
    y_total_mean = np.empty(5)
    y_pred_mean = y_pred - np.mean(y_pred,axis=0)



    for i in range(5):
        y_true_sign = np.sign(y_true[:,i])
        y_pred_sign = np.sign(y_pred[:,i])

        y_total_sign = np.multiply(y_true_sign,y_pred_sign)

        y_total[i] = np.sum(y_total_sign)


        y_pred_sign_mean = np.sign(y_pred_mean[:,i])

        y_total_sign_mean = np.multiply(y_true_sign,y_pred_sign_mean)

        y_total_mean[i] = np.sum(y_total_sign_mean)




        print(f'{pipeline_args.args["data_lag"][-i-1]}h correct amount of signs is: {y_total[i]}')
        print(f'{pipeline_args.args["data_lag"][-i - 1]}h correct amount of signs with mean removal is: {y_total_mean[i]}')


def information_coefficient(y_true,y_pred,verbose=True):
    coef_r, p_r = spearmanr(y_true, y_pred)
    alpha = 0.05

    if verbose:
        if p_r < alpha:
            print('Samples are correlated (reject H0) p=%.3f' % p_r)
            print('Spearmans correlation coefficient: %.3f' % coef_r)
        else:
            print('Samples are un-correlated (Fail to reject H0) p=%.3f' % p_r)
            print('Spearmans correlation coefficient: %.3f' % coef_r)

    return coef_r,p_r

def ic_coef(y_true,y_pred):
    # if pipeline_args.args['expand_dims'] == False:
    #     y_pred = y_pred[:,-1,:]

    for i in range(5):
        print(f'{pipeline_args.args["data_lag"][-i-1]}h lag spearman statistics:')
        information_coefficient(y_true[:,i],y_pred[:,i])

'''Function that performs basic vectorized backtest.

Lag adjusts true and pred arrays based on data_lag arguments.
This way, if lag is 12 hours, it is assumed the asset was bought//short every 12 hours, not every hour.
The values are shifted so that the last point corresponds to last point in the array.
Spearmann analysis is shown for lag-adjusted arrays.
Plots 5 time lags strategy performance vs the underlying buy and hold strategy.
Accepts: 5 dimensional y_true and y_pred numpy arrays.'''

def vectorized_backtest(y_true_input,y_pred_input):
    #ic_coef(y_true_input,y_pred_input)

    for i in range(0,5):
        y_true = y_true_input[:,i]
        y_pred = y_pred_input[:,i]
        lag = pipeline_args.args["data_lag"][-i-1]
        y_true = pd.Series(y_true)
        #print(y_true)

        coef_r, p_r = spearmanr(y_true, y_pred)
        # if p_r > 0.05:
        #     print(f'{lag} lag is not statistically correlated')
        if coef_r < 0:
            #print(f'inverse! for {lag}lag')
            y_pred = -1*y_pred
        #print(coef_r)

        if lag != 1:
            shift = len(y_true) % lag - 1
            if shift < 0:
                shift = len(y_true) % lag + lag - 1
        else:
            shift = 1


        y_true = y_true.iloc[shift::lag]


        y_pred = pd.Series(y_pred)
        y_pred = y_pred.iloc[shift::lag]



        #print(y_pred)



        long_signals = pd.Series(np.where(y_pred>0,1,0),index=y_true.index)
        short_signals = pd.Series(np.where(y_pred<0,-1,0),index=y_true.index)

        long_returns = long_signals.mul(y_true)


        short_returns = short_signals.mul(y_true)

        print(f'for {lag} the latest long signal is:',long_signals.iloc[-1],'short signal:',short_signals.iloc[-1])

        strategy = long_returns.add(short_returns)
    # print(strategy.shape)
        strategy_cum = (1+strategy).cumprod() - 1
        y_true_cum = (1+y_true).cumprod() - 1

        ax = plt.subplot(5, 1, i + 1)
        ax.plot(strategy_cum, label='Strategy performance')
        ax.plot(y_true_cum, label='Real performance')

        plt.title(f'{pipeline_args.args["data_lag"][-i - 1]} hours lag returns')
        ax.legend()
    plt.show()





