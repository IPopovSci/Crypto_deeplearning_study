import matplotlib.pyplot as plt
import os
from Networks.network_config import NetworkParams
import numpy as np
from pipeline.pipelineargs import PipelineArgs
from dotenv import load_dotenv
from utility import remove_mean

load_dotenv()
pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()

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

    path = os.getenv("model_path")
    model_load_name = os.environ['model_load_name']
    plt.savefig(
        f'{path}/{pipeline_args.args["interval"]}/{pipeline_args.args["ticker"]}/{network_args.network["model_type"]}/{model_load_name}_graph.png')
    plt.show()