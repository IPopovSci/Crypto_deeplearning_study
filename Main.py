from yfinance_facade import ticker_data
from plotting import plot_stock
from ta_feature_add import add_ta
from PCA import pca_reduction
from tt_split import train_test_split_custom
from data_scaling import SS_transform,min_max_sc
from Arguments import args
from build_timeseries import build_timeseries
from data_trim import trim_dataset
from LSTM_network import create_lstm_model
from callbacks import mcp,custom_loss
from keras.models import Sequential, load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os
from attention import Attention

ticker = 'GME'

BATCH_SIZE = args['batch_size']
#
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
pca_reduction(ticker)
'''Step 6 - Perform MinMaxScaling'''
x_train,x_test = min_max_sc(ticker)
'''Step 7 - Create Time-series'''
y_col_index = args['n_components'] - 1 #Minus one because y_col_index searches for the next column (I.e have to indicate the previous one)
x_t, y_t = build_timeseries(x_train, y_col_index)
x_t = trim_dataset(x_t, BATCH_SIZE)
y_t = trim_dataset(y_t, BATCH_SIZE)
'''Step 8 - Initialize Model'''
lstm_model = create_lstm_model(x_t)
print(lstm_model.summary())
'''Step 9 - Break Test into test and validation'''
x_temp, y_temp = build_timeseries(x_test, y_col_index)
x_test_t, x_val = np.array_split(trim_dataset(x_temp, BATCH_SIZE), 2)
y_test_t, y_val = np.array_split(trim_dataset(y_temp, BATCH_SIZE), 2)
print("Test size", x_test_t.shape, y_test_t.shape, x_val.shape, y_val.shape)
# '''Step 10 - Fit the model'''
history_lstm = lstm_model.fit(x_t, y_t, epochs=args["epochs"], verbose=1, batch_size=BATCH_SIZE,
                              shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                                                              trim_dataset(y_val, BATCH_SIZE)),callbacks=[mcp])
'''Step 11 - Load the model and predict'''
saved_model = load_model(os.path.join('data\output', 'best_lstm_model.h5'), custom_objects={'custom_loss': custom_loss,'attention': Attention})

y_pred_lstm = saved_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
y_pred_lstm = y_pred_lstm.flatten()
y_test_t = trim_dataset(y_test_t, BATCH_SIZE)

error_lstm = mean_squared_error(y_test_t, y_pred_lstm)
# print("Error is", error_lstm, y_pred_lstm.shape, y_test_t.shape)
# print(y_pred_lstm[0:15])
# print(y_test_t[0:15])
'''Step 12 - Graph the results'''
from matplotlib import pyplot as plt
plt.figure(figsize = (12, 6))
plt.plot(y_pred_lstm)
plt.plot(y_test_t)
plt.title('Prediction vs Real Stock Price')
plt.ylabel('Price')
plt.xlabel('Days')
plt.legend(['Prediction', 'Real'], loc='best')
plt.show()