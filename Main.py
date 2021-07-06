from yfinance_facade import ticker_data
from plotting import plot_stock
from ta_feature_add import add_ta
from PCA import pca_reduction
from tt_split import train_test_split_custom
from data_scaling import SS_transform
from Arguments import args
ticker = '^GSPC'

'''Step 1 - Download stock price data from yahoo finance'''
ticker_data(ticker)

'''Step 2 - Plot stock price & volume'''
plot_stock(ticker,False)

'''Step 3 - Add TA Analysis'''
add_ta(ticker)
'''Step 4 - Split data into training set and test set'''
train_test_split_custom(ticker)
'''Step 5 - Perform StanardScaler Reduction'''
SS_transform(ticker)
'''Step 6 - Perform PCA Reduction'''
pca_reduction(ticker,30)
