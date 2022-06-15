from flask import Blueprint, render_template, request, flash, jsonify
import random

random.seed(1337)
import numpy as np

np.random.seed(1337)
import sys
from pipeline.pipelineargs import PipelineArgs
from dotenv import load_dotenv
from Networks.network_config import NetworkParams
import os
import pathlib
from pathlib import Path
from training import model_train, model_predict
from pipeline.pipeline_structure import pipeline
import sqlite3
from sqlite3 import Error
from sqlalchemy import create_engine
from . import db
from .models import Model_params
from Backtesting.Backtesting import backtest_total
from Data_Processing.data_trim import trim_dataset
import base64
from base64 import b64encode


load_dotenv()


# Defining directories to use
os.environ['mm_path'] = f'{Path(sys.path[0]).parent}/scalers'
os.environ['ss_path'] = f'{Path(sys.path[0]).parent}/scalers'
os.environ['model_path'] = f'{Path(sys.path[0]).parent}/models'
os.environ['data_path'] = f'{Path(sys.path[0]).parent}/Data'

pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()
db_path = f'{Path(sys.path[0]).parents[1]}/sql/model_params.sqlite'
# print(db_path)
# Defining internal variables

pipeline_args.args['mode'] = 'prediction'  # training or prediction

image_folder = f'{pathlib.Path(os.environ["data_path"]).parent}/flask_api/website/static/'
views = Blueprint('views', __name__)
model_path = Path(os.environ['model_path']).parents[0] / 'models'
root = Path(os.environ['model_path']).parents[0] / 'models'
intervals = [f.name for f in os.scandir(model_path) if f.is_dir()]  # so we get intervals
tickers = {}
model_types = []

'''Creates default values for dropdown menus
Returns: 4 strings, corresponding to each menu'''
def get_default_values():
    intervals = Model_params.query.with_entities(Model_params.interval).distinct()
    default_intervals = []
    default_intervals.append('Select Interval')
    for interval in intervals:
        default_intervals.append(interval[0])

    default_tickers = []
    default_tickers.append('Select Ticker')

    default_types = []
    default_types.append('Select Model Type')

    default_models = []
    default_models.append('Select Model')

    return default_intervals, default_tickers, default_types, default_models

'''Runs the main page'''
@views.route('/', methods=["GET"])
def home():
    default_intervals, default_tickers, default_model_types, default_models = get_default_values()

    return render_template('home.html', all_intervals=default_intervals, all_tickers=default_tickers,
                           all_model_types=default_model_types, all_model=default_models)

'''Updates dropdowns
Uses sqalchemy to select currently picked options from SQL database
Returns JSON for ticker,model_type and model selections'''
@views.route('/_update_dropdown')
def update_dropdown():
    selected_interval = request.args.get('selected_interval', type=str)

    selected_ticker = Model_params.query.with_entities(Model_params.ticker).filter_by(
        interval=selected_interval).distinct()

    ticker_selection = ''
    ticker_selection += '<option value="Select Ticker">Select Ticker</option>'
    for entry in selected_ticker:
        ticker_selection += '<option value="{}">{}</option>'.format(entry[0], entry[0])

    '''----------------'''

    selected_ticker = request.args.get('selected_ticker', type=str)

    selected_model_type = Model_params.query.with_entities(Model_params.type).filter_by(ticker=selected_ticker,
                                                                                        interval=selected_interval).distinct()

    model_type_selection = ''
    model_type_selection += '<option value="Select Model Type">Select Model Type</option>'
    for entry in selected_model_type:
        model_type_selection += '<option value="{}">{}</option>'.format(entry[0], entry[0])

    '''-----------------'''

    selected_model_type = request.args.get('selected_model_type', type=str)

    selected_model = Model_params.query.with_entities(Model_params.model_name).filter_by(type=selected_model_type,
                                                                                         ticker=selected_ticker,
                                                                                         interval=selected_interval).distinct()

    model_selection = ''
    for entry in selected_model:
        model_selection += '<option value="{}">{}</option>'.format(entry[0], entry[0])

    return jsonify(ticker_selection=ticker_selection, model_type_selection=model_type_selection,
                   model_selection=model_selection)

'''Processes the button press
Acquires currently picked values from dropdown menus
Sets the relevant parameters
Loads in the data
Performs prediction and basic backtesting
loads in backtest images
Returns: JSON with 5 time lags: Predictions, Spearmanns coefficient and 2 images'''
@views.route('/_process_data', methods=['POST'])
def process_data():
    backtest = request.form.get('backtest')

    os.environ['ensemble'] = 'NotAvg' #This is a reminder to add ensemble, and to activate backtest invert function in case ensembly=on


    pipeline_args.args['interval'] = request.form.get('Interval')

    pipeline_args.args['ticker'] = request.form.get('Ticker', type=str)

    network_args.network["model_type"] = request.form.get('Model_Type', type=str)

    selected_model = request.form.get('Model', type=str)

    input_shape = Model_params.query.with_entities(Model_params.input_shape).filter_by(model_name=selected_model).one()
    pipeline_args.args['batch_size'] = int(input_shape[0].split(',')[0].strip('()'))

    pipeline_args.args['time_steps'] = int(input_shape[0].split(',')[1].strip('()'))

    # pipeline_args.args['cryptowatch_key'] = os.environ['cryptowatch_key'] #If we want to use custom key here

    model_load_name = selected_model
    os.environ['model_load_name'] = model_load_name

    if network_args.network["model_type"] == 'conv2d' or network_args.network["model_type"] == 'convlstm':
        pipeline_args.args['expand_dims'] = True
    else:
        pipeline_args.args['expand_dims'] = False

    x_t, y_t, x_val, y_val, x_test_t, y_test_t, size = pipeline()

    y_pred = model_predict.predict(x_test_t, f'{model_load_name}')

    if pipeline_args.args['expand_dims'] == False:
        y_pred = y_pred[:, -1, :]  # Because Dense predictions will have timesteps
    if backtest == 'True':
        y_pred_mean,ic_coef_hist,y_total_mean = backtest_total(trim_dataset(y_test_t, pipeline_args.args['batch_size']), y_pred, plot_mean=True,
                                      backtest_mean=True)

        with open(f"{image_folder}/{model_load_name}_backtest.png", "rb") as f:
            image_backtest = f.read()
        image_backtest = b64encode(image_backtest).decode("utf-8")
        with open(f"{image_folder}/{model_load_name}_graph.png", "rb") as f:
            image_graph = f.read()
        image_graph = b64encode(image_graph).decode("utf-8")
        return jsonify(preds1h="{}".format(y_pred_mean[-1, :][0]), preds4h='{}'.format(y_pred_mean[-1, :][1]),
                       preds12h='{}'.format(y_pred_mean[-1, :][2]), preds24h='{}'.format(y_pred_mean[-1, :][3]),
                       preds48h='{}'.format(y_pred_mean[-1, :][4]), sc1h="{}".format(ic_coef_hist[0]),
                       sc4h="{}".format(ic_coef_hist[1]), sc12h="{}".format(ic_coef_hist[2]),
                       sc24h="{}".format(ic_coef_hist[3]), sc48h="{}".format(ic_coef_hist[4]),
                       image_backtest=image_backtest,image_graph=image_graph)
