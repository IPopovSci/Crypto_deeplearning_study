
from pipeline.pipeline_structure import pipeline
import sys
from pipeline.pipelineargs import PipelineArgs
from dotenv import load_dotenv
from Networks.network_config import NetworkParams
from Backtesting.Backtesting import correct_signs, ic_coef
from plotting import plot_results_v2, plot_ic
from utility import remove_mean,remove_std
from Backtesting.Backtesting import vectorized_backtest
from training import model_train,model_predict
import os

load_dotenv()

os.environ['mm_path'] = f'{sys.path[0]}/scalers'
os.environ['ss_path'] = f'{sys.path[0]}/scalers'
os.environ['model_path'] = f'{sys.path[0]}/models'
os.environ['data_path'] = f'{sys.path[0]}/Data'



pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()

batch_size = os.environ['batch_size']
time_steps = os.environ['time_steps']

pipeline_args.args['mode'] =os.environ['mode'] #training or prediction
#pipeline_args.args['mode'] = 'prediction' #training or prediction
#pipeline_args.args['mode'] = 'continue' #training or prediction

pipeline_args.args['time_steps'] = int(time_steps) #1 for dense
network_args.network["model_type"] = os.environ['model_type']
print(time_steps)
model_load_name = os.environ['model_load_name']

if network_args.network["model_type"] == 'conv2d' or network_args.network["model_type"] == 'convlstm':
    pipeline_args.args['expand_dims'] = True



x_t, y_t, x_val, y_val, x_test_t, y_test_t, size = pipeline()

if pipeline_args.args['mode'] == 'training':
    model_train.train_model(x_t, y_t, x_val, y_val, network_args.network["model_type"])
elif pipeline_args.args['mode'] == 'prediction':
    y_pred = model_predict.predict(x_test_t, y_test_t, f'{model_load_name}')

    '12h:-'
    '24h:+'
    '48h:++-'
    #if you want pretty graphs and some backtests:

    if pipeline_args.args['expand_dims'] == False:
        y_pred = y_pred[:, -1, :]

    y_pred_mean = remove_mean(y_pred)
    y_test_t_mean = remove_mean(y_test_t)

    #y_pred_mean = 0.1 * y_pred_mean
    print(y_pred[-1])


    ic_coef(y_test_t, y_pred_mean)
    plot_results_v2(y_test_t, y_pred, no_mean=False)
    correct_signs(y_test_t,y_pred)
    vectorized_backtest(y_test_t, y_pred)
else:
    model_train.continue_training(x_t, y_t, x_val, y_val, f'{model_load_name}')

