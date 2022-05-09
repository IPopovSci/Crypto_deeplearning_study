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
from sqlalchemy import  create_engine
from . import db
from .models import Model_params

# db_path = f'{Path(sys.path[0]).parents[0]}/sql/model_params.sqlite'
# print(db_path)
# engine = create_engine(f'sqlite:///{db_path}')
#
# engine.connect()
#
# print(engine)
# models = Model_params.query('type').all()
#
# for model in models:
#     print(model)

load_dotenv()

#Write an init that initializes those + creates folders mb for api later
# Defining directories to use ( TODO: Wrap this into a function + folder creation)
os.environ['mm_path'] = f'{Path(sys.path[0]).parent}/scalers'
os.environ['ss_path'] = f'{Path(sys.path[0]).parent}/scalers'
os.environ['model_path'] = f'{Path(sys.path[0]).parent}/models'
os.environ['data_path'] = f'{Path(sys.path[0]).parent}/Data'

pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()
db_path = f'{Path(sys.path[0]).parents[1]}/sql/model_params.sqlite'
#print(db_path)
# Defining internal variables
pipeline_args.args['batch_size'] = int(os.environ['batch_size'])
pipeline_args.args['mode'] = 'prediction'  # training or prediction
pipeline_args.args['time_steps'] = int(os.environ['time_steps'])  # 1 for dense
network_args.network["model_type"] = os.environ['model_type']
model_load_name = os.environ['model_load_name']
pipeline_args.args['ticker'] = os.environ['ticker']
pipeline_args.args['interval'] = os.environ['interval']
pipeline_args.args['cryptowatch_key'] = os.environ['cryptowatch_key']
os.environ['TF_DETERMINISTIC_OPS'] = '1'




views = Blueprint('views',__name__)
model_path = Path(os.environ['model_path']).parents[0] / 'models'
root = Path(os.environ['model_path']).parents[0] / 'models'
intervals = [f.name for f in os.scandir(model_path) if f.is_dir()] #so we get intervals
tickers = {}
model_types = []



def get_default_values():
    intervals = Model_params.query.with_entities(Model_params.interval).distinct()
    default_intervals = []
    default_intervals.append('Select Interval')
    for interval in intervals:
        default_intervals.append(interval[0])


    tickers = Model_params.query.with_entities(Model_params.ticker).filter_by(interval=default_intervals[0]).distinct()
    default_tickers = []
    default_tickers.append('Select Ticker')
    for ticker in tickers:
        default_tickers.append(ticker[0])


    model_types = Model_params.query.with_entities(Model_params.type).filter_by(ticker=default_tickers[0]).distinct()
    default_types = []
    default_types.append('Select Model Type')

    for type in model_types:
        default_types.append(type[0])


    models = Model_params.query.with_entities(Model_params.model_name).filter_by(type=default_types[0]).distinct()
    default_models = []
    default_types.append('Select Model')
    for model in models:
        default_models.append(model[0])


    return default_intervals,default_tickers,default_types,default_models

@views.route('/', methods=["GET", "POST"])
def home():
    default_intervals,default_tickers,default_model_types,default_model = get_default_values()

    return render_template('home.html',all_intervals= default_intervals,all_tickers=default_tickers,all_model_types=default_model_types,all_model=default_model)


@views.route('/_update_dropdown')
def update_dropdown():

    selected_interval = request.args.get('selected_interval', type=str)

    print('selected interval',selected_interval)
    selected_ticker = Model_params.query.with_entities(Model_params.ticker).filter_by(interval=selected_interval).distinct()

    ticker_selection = ''
    ticker_selection += '<option value="Select Ticker">Select Ticker</option>'
    for entry in selected_ticker:
        ticker_selection += '<option value="{}">{}</option>'.format(entry[0], entry[0])

    '''----------------'''

    selected_ticker = request.args.get('selected_ticker', type=str)

    selected_model_type = Model_params.query.with_entities(Model_params.type).filter_by(ticker=selected_ticker,interval=selected_interval).distinct()

    model_type_selection = ''
    model_type_selection += '<option value="Select Model Type">Select Model Type</option>'
    for entry in selected_model_type:
        model_type_selection += '<option value="{}">{}</option>'.format(entry[0], entry[0])


    '''-----------------'''


    selected_model_type = request.args.get('selected_model_type', type=str)


    selected_model = Model_params.query.with_entities(Model_params.model_name).filter_by(type=selected_model_type,ticker=selected_ticker,interval=selected_interval).distinct()

    #print(selected_model)
    model_selection = ''
    for entry in selected_model:

        model_selection += '<option value="{}">{}</option>'.format(entry[0], entry[0])





    return jsonify(ticker_selection=ticker_selection,model_type_selection=model_type_selection,model_selection=model_selection)

@views.route('/_process_data')
def process_data():
    #print('process data triggered')
    selected_interval = request.args.get('selected_interval', type=str)
    #print('interval selected')

    pipeline_args.args['interval'] = selected_interval

    selected_ticker = request.args.get('selected_ticker', type=str)

    pipeline_args.args['ticker'] = selected_ticker

    selected_model_type = request.args.get('selected_model_type', type=str)

    network_args.network["model_type"] = selected_model_type


    selected_model = request.args.get('selected_model',type=str)


    model_load_name = selected_model

    # x_t, y_t, x_val, y_val, x_test_t, y_test_t, size = pipeline()
    #
    # y_pred = model_predict.predict(x_test_t[:-1], f'{model_load_name}')

    return None#jsonify(random_text="preds for today are {}".format(y_pred[-1,-1,:]))
