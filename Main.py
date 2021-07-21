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
from build_timeseries import build_timeseries



ticker = args['ticker']
BATCH_SIZE = args['batch_size']

"""Load Data and prep"""
x_t,y_t,x_val,y_val,x_test_t,y_test_t = data_prep(ticker)
