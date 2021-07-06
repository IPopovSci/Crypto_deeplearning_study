import matplotlib.pyplot as plt
import pandas as pd

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