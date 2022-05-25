from pipeline.pipelineargs import PipelineArgs
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from plotting import plot_results_v2
import os
from Networks.network_config import NetworkParams
from scipy.stats import spearmanr
from utility import remove_mean
import matplotlib.pyplot as plt
import pathlib

load_dotenv()
pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()

'''Function that calculates the amount of equal signs between y_true and y_pred
Accepts: y_true,y_pred.
Returns: Print out of the information.'''


def correct_signs(y_true, y_pred):
    y_total = np.empty(5)
    y_total_mean = np.empty(5)
    y_pred_mean = y_pred - np.mean(y_pred, axis=0)

    for i in range(5):
        y_true_sign = np.sign(y_true[:, i])
        y_pred_sign = np.sign(y_pred[:, i])

        y_total_sign = np.multiply(y_true_sign, y_pred_sign)

        y_total[i] = np.sum(y_total_sign)

        y_pred_sign_mean = np.sign(y_pred_mean[:, i])

        y_total_sign_mean = np.multiply(y_true_sign, y_pred_sign_mean)

        y_total_mean[i] = np.sum(y_total_sign_mean)

        print(f'{pipeline_args.args["data_lag"][-i - 1]}h correct amount of signs is: {y_total[i]}')
        print(
            f'{pipeline_args.args["data_lag"][-i - 1]}h correct amount of signs with mean removal is: {y_total_mean[i]}')

    return y_pred_mean,y_total_mean


def information_coefficient(y_true, y_pred, verbose=True):
    coef_r, p_r = spearmanr(y_true, y_pred)
    alpha = 0.05

    if verbose:
        if p_r < alpha:
            print('Samples are correlated (reject H0) p=%.3f' % p_r)
            print('Spearmans correlation coefficient: %.3f' % coef_r)
        else:
            print('Samples are un-correlated (Fail to reject H0) p=%.3f' % p_r)
            print('Spearmans correlation coefficient: %.3f' % coef_r)

    return coef_r, p_r


def ic_coef(y_true, y_pred):
    ic_coef_stor = []
    for i in range(5):
        print(f'{pipeline_args.args["data_lag"][-i - 1]}h lag spearman statistics:')
        coef_r, _ =information_coefficient(y_true[:, i], y_pred[:, i])
        ic_coef_stor.append(coef_r)
    return ic_coef_stor

'''Vector backtest that incorporates simulated trading every hour
Accepts: y_true - nx5 array of real cumulative returns
         y_pred - nx5 array of predicted cumulative returns
         balance - initial balance (float)
         lag - current lag for calculation (int)
         bet_amount - fixed bet size (float)
Returns: nx1 array with the simulated account balance'''
def advanced_vector_backtest(y_true, y_pred, balance, lag, bet_amount):
    balance_tab = np.full([y_true.shape[0], 1], balance)
    for tick in range(0, y_pred.shape[0]):
        if tick + lag < y_pred.shape[0]:  # Prevent open positions before the lag end time
            if tick < lag:  # position open only
                balance_tab[tick + 1] = balance_tab[tick] - bet_amount
            elif tick < y_pred.shape[0] - lag - 1:  # trading
                profit = bet_amount * (1. + 100. * y_true[tick] * np.sign(y_pred[tick]))
                balance_tab[tick + 1] = balance_tab[tick] - bet_amount + profit
        else:  # we don't know the future returns
            balance_tab[tick] = balance_tab[tick - 1]
    return balance_tab

'''Buy and hold strategy value calculator
Accepts: y_true - nx5 array of real cumulative returns
         y_pred - nx5 array of predicted cumulative returns
         balance - initial balance (float)
Returns: nx1 array with simulated account balance'''
def buy_hold(y_true, y_pred, balance):
    lag = 1
    balance_tab = np.full([y_true.shape[0], 1], balance)
    for tick in range(0, y_pred.shape[0]):
        if tick + lag < y_pred.shape[0]:  # Prevent open positions before the lag end time
            if tick < lag:  # position open only
                balance_tab[tick + 1] = balance_tab[tick]
            elif tick < y_pred.shape[0] - lag:  # trading
                balance_tab[tick + 1] = balance_tab[tick] * (1 + y_true[tick])
        else:  # we don't know the future do we
            balance_tab[tick] = balance_tab[tick - 1]
    return balance_tab


'''Function that performs basic vectorized backtest.

Lag adjusts true and pred arrays based on data_lag arguments.
This way, if lag is 12 hours, it is assumed the asset was bought//short every 12 hours, not every hour.
The values are shifted so that the last point corresponds to last point in the array.
Spearmann analysis is shown for lag-adjusted arrays.
Plots 5 time lags strategy performance vs the underlying buy and hold strategy.
Accepts: 5 dimensional y_true and y_pred numpy arrays.'''


def vectorized_backtest(y_true_input, y_pred_input):
    plt.figure(figsize=(15, 10))
    for i in range(0, 5):
        y_true = y_true_input[:, i]
        y_pred = y_pred_input[:, i]
        lag = pipeline_args.args["data_lag"][-i - 1]
        lag_store = lag

        coef_r, p_r = spearmanr(y_true_input[:, i], y_pred)
        if os.environ['ensemble'] != 'average':
            if coef_r < 0:  # we only need this when not doing ensembly (ensembly will flip auto)
                print(f'inverse! for {lag}lag')
                y_pred = -1 * y_pred

        balance = advanced_vector_backtest(y_true, y_pred, 10000., lag, 100.)

        y_pred = pd.Series(y_pred)
        y_true = pd.Series(y_true)

        long_signals = pd.Series(np.where(y_pred > 0, 1, 0), index=y_true.index)
        short_signals = pd.Series(np.where(y_pred < 0, -1, 0), index=y_true.index)

        print(f'for {lag_store} the latest long signal is:', long_signals.iloc[-1], 'short signal:',
              short_signals.iloc[-1])

        buy_hold_hist = buy_hold(y_true_input[:, 0], y_pred, 10000.)

        ax = plt.subplot(5, 1, i + 1)
        ax.plot(balance, label='Strategy performance')
        ax.plot(buy_hold_hist, label='Real performance')

        plt.title(f'{pipeline_args.args["data_lag"][-i - 1]} hours lag returns')
        ax.legend(loc=3)

    path = os.getenv("model_path")
    model_load_name = os.environ['model_load_name']
    plt.savefig(
        f'{pathlib.Path(path).parent}/flask_api/website/static/{model_load_name}_backtest.png')
    #print('I saved the backtest image! Here:',f'{pathlib.Path(path).parent}/flask_api/website/static/{model_load_name}_backtest.png' )
    #plt.show()
    plt.close()


'''This function combines the usage of both backtests above
Accepts: True values, pred values, switch to remove mean from plotting, switch to remove mean from backtesting
Prints information coefficient, plots the results of both backtests, and displays number of correct signs'''


def backtest_total(y_true, y_pred, plot_mean=True, backtest_mean=True):
    try:
        if pipeline_args.args['expand_dims'] == False:
            y_pred = y_pred[:, -1, :]
    except:
        ''

    ic_coef_hist = ic_coef(y_true, y_pred)
    plot_results_v2(y_true, y_pred, no_mean=plot_mean)
    y_pred_mean,y_total_mean = correct_signs(y_true, y_pred)

    if backtest_mean:
        vectorized_backtest(y_true, y_pred_mean)
    else:
        vectorized_backtest(y_true, y_pred)
    return y_pred_mean,ic_coef_hist,y_total_mean