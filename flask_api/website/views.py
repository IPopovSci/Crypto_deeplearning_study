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
import json

load_dotenv()

#Write an init that initializes those + creates folders mb for api later
# Defining directories to use ( TODO: Wrap this into a function + folder creation)
os.environ['mm_path'] = f'{sys.path[0]}/scalers'
os.environ['ss_path'] = f'{sys.path[0]}/scalers'
os.environ['model_path'] = f'{sys.path[0]}/models'
os.environ['data_path'] = f'{sys.path[0]}/Data'

pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()

# Defining internal variables
pipeline_args.args['batch_size'] = int(os.environ['batch_size'])
pipeline_args.args['mode'] = os.environ['mode']  # training or prediction
pipeline_args.args['time_steps'] = int(os.environ['time_steps'])  # 1 for dense
network_args.network["model_type"] = os.environ['model_type']
model_load_name = os.environ['model_load_name']
pipeline_args.args['ticker'] = os.environ['ticker']
pipeline_args.args['interval'] = os.environ['interval']
pipeline_args.args['cryptowatch_key'] = os.environ['cryptowatch_key']




views = Blueprint('views',__name__)
model_path = Path(os.environ['model_path']).parents[1] / 'models'
root = Path(os.environ['model_path']).parents[1] / 'models'
intervals = [f.name for f in os.scandir(model_path) if f.is_dir()] #so we get intervals
tickers = {}
model_types = []

def get_folder_layer(path):
    my_dict = {}
    for f in os.scandir(path):
        if f.is_dir():
            my_dict[f.name] = get_folder_layer(f)
        else:
            my_dict[f.name] = ""
    return my_dict



# returns dict of dicts of dicts etc.
def get_dropdown_values(root):
    class_entry_relations = get_folder_layer(root)
    return class_entry_relations


@views.route('/_get_struct')
def get_struct():
    my_struct = get_dropdown_values(root)
    print("Returning folder structure: " + my_struct)
    return jsonify(my_struct)


@views.route('/_process_data')
def process_data():
    #print('process data triggered')
    selected_interval = request.args.get('selected_interval', type=str)
    #print('interval selected')
    selected_ticker = request.args.get('selected_ticker', type=str)
    selected_model_type = request.args.get('selected_model_type', type=str)
    selected_model = request.args.get('selected_model',type=str)
    #print('button works')
    # process the two selected values here and return the response; here we just create a dummy string

    return jsonify(random_text="you selected {} and {} and {} and {}".format(selected_interval, selected_ticker,selected_model_type,selected_model))



@views.route('/', methods=["GET", "POST"])
def home():
    class_entry_relations = get_dropdown_values(root)
    #print(class_entry_relations)

    default_intervals = sorted(class_entry_relations.keys())
    #print(default_intervals)
    default_tickers = sorted(class_entry_relations[default_intervals[0]])
    #print(default_tickers)

    default_model_types = sorted(class_entry_relations[default_intervals[0]][default_tickers[0]])

    #print(default_model_types)

    default_model = sorted(class_entry_relations[default_intervals[0]][default_tickers[0]][default_model_types[0]])

    #print(default_model)

    return render_template('home.html',all_intervals= default_intervals,all_tickers=default_tickers,all_model_types=default_model_types,all_model=default_model)


selected_interval_store = sorted(get_dropdown_values(root).keys())[0]
selected_ticker_store = sorted(get_dropdown_values(root)[selected_interval_store])[0]

@views.route('/_update_dropdown_model_type')
def update_dropdown_model_type():
    selected_interval = request.args.get('selected_interval', type=str)
    print('selected_interval', selected_interval)
    selected_ticker = request.args.get('selected_ticker', type=str)
    print('selected_ticker', selected_ticker)
    selected_model_type = get_dropdown_values(root)[selected_interval][selected_ticker]
    #print(selected_model_type)
    # create the value sin the dropdown as a html string
    model_type_selection = ''
    for entry in selected_model_type:
        print(entry)
        model_type_selection += '<option value="{}">{}</option>'.format(entry, entry)


    return jsonify(model_type_selection=model_type_selection)

@views.route('/_update_dropdown_model')
def update_dropdown_model_types():
    selected_interval = request.args.get('selected_interval', type=str)
    print('selected_interval', selected_interval)
    selected_ticker = request.args.get('selected_ticker', type=str)
    print('selected_ticker', selected_ticker)
    selected_model_type = request.args.get('selected_model_type', type=str)

    selected_model = get_dropdown_values(root)[selected_interval][selected_ticker][selected_model_type]
    print(selected_model)
    model_selection = ''
    for entry in selected_model:
        model_selection += '<option value="{}">{}</option>'.format(entry, entry)


    return jsonify(model_type=model_selection)

@views.route('/_update_dropdown')
def update_dropdown():
    global selected_interval_store
    global selected_ticker_store
    # the value of the first dropdown (selected by the user)
    selected_interval = request.args.get('selected_interval', type=str)
    # get values for the second dropdown
    print('selected interval',selected_interval)
    selected_ticker = get_dropdown_values(root)[selected_interval]
    print('selected_ticker',selected_ticker)



    #print('selected ticker keys', selected_ticker.keys())
    # create the value sin the dropdown as a html string
    ticker_selection = ''
    for entry in selected_ticker.keys():
        ticker_selection += '<option value="{}">{}</option>'.format(entry, entry)





    # selected_model_type = request.args.get('selected_model_type', type=str)
    #
    # # get values for model
    #
    # selected_model = get_dropdown_values(root)[selected_interval][selected_ticker][selected_model_type]
    # print(selected_model)
    # model_selection = ''
    # for entry in selected_model:
    #     model_selection += '<option value="{}">{}</option>'.format(entry, entry)



    return jsonify(ticker_selection=ticker_selection)#,model_type_selection=model_type_selection)#,model_selection=model_selection)

