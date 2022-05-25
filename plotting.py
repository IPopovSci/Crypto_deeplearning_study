import matplotlib.pyplot as plt
import os
from Networks.network_config import NetworkParams
import numpy as np
from pipeline.pipelineargs import PipelineArgs
from dotenv import load_dotenv
from utility import remove_mean
import pandas as pd
import pathlib

load_dotenv()
pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()

'''This function plots the 5 time lags using subplots.
Accepts: array with true values, array with prediction values and boolean switch that removes mean if activated.
Returns: graphical plot of 5 predictions'''


def plot_results_v2(y_true_input, y_pred_input, no_mean=True):
    plt.figure(figsize=(15, 10))
    if no_mean:
        y_pred_input = remove_mean(y_pred_input)
    for i in range(5):
        ax = plt.subplot(5, 1, i + 1)
        y_true = pd.Series(y_true_input[:,i])
        y_pred = pd.Series(y_pred_input[:,i])

        ax.plot(y_true, label='True Values')
        ax.plot(y_pred, label='Pred Values')



        plt.title(f'{pipeline_args.args["data_lag"][-i - 1]} hours lag returns')
        ax.legend(loc=3)
    path = os.getenv("model_path")
    model_load_name = os.environ['model_load_name']
    plt.savefig(
        f'{pathlib.Path(path).parent}/flask_api/website/static/{model_load_name}_graph.png')
    print("I saved the graph image!")
    #plt.show()
    plt.close()