import matplotlib.pyplot as plt

import numpy as np
from pipeline.pipelineargs import PipelineArgs
from dotenv import load_dotenv
from utility import remove_mean
from Backtesting.Backtesting import information_coefficient

load_dotenv()
pipeline_args = PipelineArgs.get_instance()

'''This function plots the 5 time lags using subplots.
Accepts: array with true values, array with prediction values and boolean switch that removes mean if activated.
Returns: graphical plot of 5 predictions'''


def plot_results_v2(y_true, y_pred, no_mean=True):
    if no_mean:
        y_pred = remove_mean(y_pred)
    for i in range(5):
        ax = plt.subplot(5, 1, i + 1)
        ax.plot(y_true[:, i], label='True Values')
        ax.plot(y_pred[:, i], label='Pred Values')

        plt.title(f'{pipeline_args.args["data_lag"][-i - 1]} hours lag returns')
        ax.legend()

    plt.show()


'''This function plots the information coefficient
Currently not very useful due to lack of investment universe'''


def plot_ic(y_true, y_pred):
    coef_list = []
    p_r_list = []
    for i in range(len(y_true)):
        coef, p_r = information_coefficient(y_true[i], y_pred[i], verbose=False)
        coef_list.append(coef)
        p_r_list.append(p_r)

    mean_coef = np.mean(coef_list[:-1])
    mean_p_r = np.mean(p_r_list[:-1])

    plt.figure(figsize=(12, 6))
    plt.plot(coef_list)
    plt.axhline(y=mean_coef, color='r', linestyle='-')
    plt.axhline(y=mean_p_r, color='y', linestyle='-')
    plt.show()
