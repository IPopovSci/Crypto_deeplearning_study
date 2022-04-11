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
    for i in range(5):
        print(f'{pipeline_args.args["data_lag"][-i - 1]}h lag spearman statistics:')
        information_coefficient(y_true[:, i], y_pred[:, i])


'''Function that performs basic vectorized backtest.

Lag adjusts true and pred arrays based on data_lag arguments.
This way, if lag is 12 hours, it is assumed the asset was bought//short every 12 hours, not every hour.
The values are shifted so that the last point corresponds to last point in the array.
Spearmann analysis is shown for lag-adjusted arrays.
Plots 5 time lags strategy performance vs the underlying buy and hold strategy.
Accepts: 5 dimensional y_true and y_pred numpy arrays.'''


def vectorized_backtest(y_true_input, y_pred_input):
    for i in range(0, 5):
        y_true = y_true_input[:, 0]

        y_pred = y_pred_input[:, i]
        lag = pipeline_args.args["data_lag"][-i - 1]
        lag_store = lag
        y_true = pd.Series(y_true)
        # print(y_true)

        coef_r, p_r = spearmanr(y_true_input[:, i], y_pred)

        if coef_r < 0:
            print(f'inverse! for {lag}lag')
            y_pred = -1 * y_pred

        y_pred = pd.Series(y_pred)

        long_signals = pd.Series(np.where(y_pred > 0, 1, 0), index=y_true.index)
        short_signals = pd.Series(np.where(y_pred < 0, -1, 0), index=y_true.index)

        long_returns = long_signals.mul(y_true)

        short_returns = short_signals.mul(y_true)

        print(f'for {lag_store} the latest long signal is:', long_signals.iloc[-1], 'short signal:',
              short_signals.iloc[-1])

        strategy = long_returns.add(short_returns)
        strategy_cum = (1 + strategy).cumprod() - 1
        y_true_cum = (1 + y_true).cumprod() - 1

        strategy_cum_rolling = strategy_cum.rolling(lag).mean()
        y_true_cum_rolling = y_true_cum.rolling(lag).mean()

        ax = plt.subplot(5, 1, i + 1)
        ax.plot(strategy_cum_rolling, label='Strategy performance')
        ax.plot(y_true_cum_rolling, label='Real performance')

        plt.title(f'{pipeline_args.args["data_lag"][-i - 1]} hours lag returns')
        ax.legend()

    path = os.getenv("model_path")
    model_load_name = os.environ['model_load_name']
    plt.savefig(
        f'{path}/{pipeline_args.args["interval"]}/{pipeline_args.args["ticker"]}/{network_args.network["model_type"]}/{model_load_name}_backtest.png')
    plt.show()


'''This function combines the usage of both backtests above
Accepts: True values, pred values, switch to remove mean from plotting, switch to remove mean from backtesting
Prints information coefficient, plots the results of both backtests, and displays number of correct signs'''
def backtest_total(y_true, y_pred, plot_mean=True, backtest_mean=False):
    if pipeline_args.args['expand_dims'] == False:
        y_pred = y_pred[:, -1, :]

    y_pred_mean = remove_mean(y_pred)

    print(y_pred[-1])

    ic_coef(y_true, y_pred)
    plot_results_v2(y_true, y_pred, no_mean=plot_mean)
    correct_signs(y_true, y_pred)

    if backtest_mean:
        vectorized_backtest(y_true, y_pred_mean)
    else:
        vectorized_backtest(y_true, y_pred)
