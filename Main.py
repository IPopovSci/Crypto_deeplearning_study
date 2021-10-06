from Arguments import args
from yfinance_facade import ticker_data,vix_data
from ta_feature_add import add_ta


ticker = args['ticker']
BATCH_SIZE = args['batch_size']
start_date = args['starting_date']

'''Step 1: Get Data'''
ticker_history = ticker_data(ticker,start_date)
vix_history = vix_data(start_date)
'''Step 2: Apply TA Analysis'''
ta_data = add_ta(ticker_history,ticker)
print(ta_data)
