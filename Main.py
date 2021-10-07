from Arguments import args
from yfinance_facade import ticker_data,vix_data
from ta_feature_add import add_ta
from Detrending import test_stationarity,row_difference,inverse_difference
import matplotlib.pyplot as plt

ticker = args['ticker']
BATCH_SIZE = args['batch_size']
start_date = args['starting_date']

'''Step 1: Get Data'''
ticker_history = ticker_data(ticker,start_date)
vix_history = vix_data(start_date)
'''Step 2: Apply TA Analysis'''
ta_data = add_ta(ticker_history,ticker) #The columns names can be acessed from arguments 'train_cols'
print(ta_data['Close'])
'''Step 3: Detrend the data'''
one_day_detrend = row_difference(ta_data) #Need to modify this function to loop over columns, at least the Close, Open,etc ones
