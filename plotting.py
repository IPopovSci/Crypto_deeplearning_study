import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pipeline.pipelineargs import PipelineArgs
from dotenv import load_dotenv
from utility import remove_mean
load_dotenv()
pipeline_args = PipelineArgs.get_instance()

def plot_stock(ticker,plot=True):
    stock_data = pd.read_csv(f'data/01_seed/{ticker}.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # stock price
    ax1.set_title('HSBC stock price', fontsize=10)
    ax1.plot(stock_data['Close'])
    # volume
    ax2.set_title('HSBC stock volume', fontsize=10)
    ax2.plot(stock_data["Volume"])

    if plot == True:
        plt.show()

def plot_results(y_pred,y_test,remove_mean=True,normalize=True):
    if remove_mean:
        y_pred = y_pred - np.mean(y_pred)
    if normalize:
        scaler = np.mean(np.mean(y_test)/np.mean(y_pred))
        y_pred = y_pred / scaler
    from matplotlib import pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(y_pred)
    plt.plot(y_test)
    plt.title('Prediction vs Real Stock Price')
    plt.ylabel('Price')
    plt.xlabel('Days')
    plt.legend(['Prediction', 'Real'], loc='best')
    plt.show()

def plot_results_v2(y_true,y_pred,no_mean=True):
    if no_mean:
        y_pred = remove_mean(y_pred)
    for i in range(5):
        x = np.empty(len(y_pred))
        ax = plt.subplot(5, 1, i+1)
        ax.plot(y_true[:,i],label='True Values')
        ax.plot(y_pred[:,i],label='Pred Values')

        plt.title(f'{pipeline_args.args["data_lag"][-i - 1]} hours lag returns')
        ax.legend()

    plt.show()
