from pipeline.pipeline_structure import pipeline
import sys
from pipeline.pipelineargs import PipelineArgs
from dotenv import load_dotenv
from Networks.network_config import NetworkParams
from training import model_train, model_predict
from Backtesting.Backtesting import backtest_total
import os

load_dotenv()

os.environ['mm_path'] = f'{sys.path[0]}/scalers'
os.environ['ss_path'] = f'{sys.path[0]}/scalers'
os.environ['model_path'] = f'{sys.path[0]}/models'
os.environ['data_path'] = f'{sys.path[0]}/Data'

pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()

pipeline_args.args['batch_size'] = int(os.environ['batch_size'])
pipeline_args.args['mode'] = os.environ['mode']  # training or prediction
pipeline_args.args['time_steps'] = int(os.environ['time_steps'])  # 1 for dense
network_args.network["model_type"] = os.environ['model_type']
model_load_name = os.environ['model_load_name']
pipeline_args.args['ticker'] = os.environ['ticker']
pipeline_args.args['interval'] = os.environ['interval']

if network_args.network["model_type"] == 'conv2d' or network_args.network["model_type"] == 'convlstm':
    pipeline_args.args['expand_dims'] = True

x_t, y_t, x_val, y_val, x_test_t, y_test_t, size = pipeline()


if pipeline_args.args['mode'] == 'training':
    model_train.train_model(x_t, y_t, x_val, y_val, network_args.network["model_type"])
elif pipeline_args.args['mode'] == 'prediction':
    y_pred = model_predict.predict(x_test_t, y_test_t, f'{model_load_name}')
    backtest_total(y_test_t,y_pred,plot_mean=False,backtest_mean=False)
else:
    model_train.continue_training(x_t, y_t, x_val, y_val, f'{model_load_name}')
