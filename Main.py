from Arguments import args
from data_trim import trim_dataset
from callbacks import mcp,custom_loss
from keras.models import load_model
from sklearn.metrics import mean_squared_error
import os
from attention import Attention
from data_scaling import unscale_data
from run_functions import data_prep
from plotting import plot_results


ticker = 'GME'
BATCH_SIZE = args['batch_size']

"""Load Data and prep"""
x_t,y_t,x_val,y_val,x_test_t,y_test_t,lstm_model = data_prep(ticker)
# '''Step 10 - Fit the model'''
history_lstm = lstm_model.fit(x_t, y_t, epochs=args["epochs"], verbose=1, batch_size=BATCH_SIZE,
                              shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                                                              trim_dataset(y_val, BATCH_SIZE)),callbacks=[mcp])
'''Step 11 - Load the model and predict'''
saved_model = load_model(os.path.join('data\output', 'best_lstm_model.h5'), custom_objects={'custom_loss': custom_loss,'attention': Attention})

y_pred_lstm = saved_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
y_pred_lstm = y_pred_lstm.flatten()

y_test_t = trim_dataset(y_test_t, BATCH_SIZE)

'''Step 12 - Revert the values to real (Scalers)'''
y_pred,y_test = unscale_data(ticker,y_pred_lstm,y_test_t)



error_lstm = mean_squared_error(y_test_t, y_pred_lstm)
# print("Error is", error_lstm, y_pred_lstm.shape, y_test_t.shape)
# print(y_pred_lstm[0:15])
# print(y_test_t[0:15])
'''Step 12 - Graph the results'''
plot_results(y_pred,y_test)