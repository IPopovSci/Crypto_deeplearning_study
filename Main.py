from Arguments import args
from yfinance_facade import ticker_data, aux_data
from ta_feature_add import add_ta
from Detrending import test_stationarity, row_difference, inverse_difference
import matplotlib.pyplot as plt
from tt_split import train_test_split_custom

ticker = args['ticker']
BATCH_SIZE = args['batch_size']
start_date = args['starting_date']

'''Step 1: Get Data'''
ticker_history = ticker_data(ticker, start_date)
aux_history = aux_data(ticker_history,['CL=F','GC=F','^VIX','^TNX'], start_date) #Get any extra data
'''Step 2: Apply TA Analysis'''
ta_data = add_ta(aux_history, ticker)  # The columns names can be acessed from arguments 'train_cols'
'''Step 3: Detrend the data'''
one_day_detrend = row_difference(ta_data)
'''Step 4: Split data into training/testing'''
x_train, x_test = train_test_split_custom(one_day_detrend) #doesn't work currently
'''Step 5: SS Transform'''

'''From Paper what good practices should be: http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
Shuffle the examples - Feed various companies/markets to the training
Centre the input variables by substracting the mean - SS transform
Normalize input variables to a standard deviation of 1 - SS transform?
If possible, decorrelate input variables - PCA (Look into whiten = true in sklearn library)?
Pick a network with sigmoid function (Fig 4 in paper) - Use sigmoid activation
Set the target values within range of the sigmoid, typically -1 to 1 - Minmax scaler
Initialize weights to random values - (Maybe add a genetic algo too for mutating weights?)
'''
