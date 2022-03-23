
from pipeline.pipeline_structure import pipeline

from pipeline.pipelineargs import PipelineArgs
from dotenv import load_dotenv
from Networks.network_config import NetworkParams
from Backtesting.Backtesting import correct_signs, ic_coef
from plotting import plot_results_v2, plot_ic
from utility import remove_mean,remove_std
from Backtesting.Backtesting import vectorized_backtest
from training import model_train,model_predict

load_dotenv()

pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()

batch_size = pipeline_args.args['batch_size']
time_steps = pipeline_args.args['time_steps']

pipeline_args.args['mode'] = 'prediction' #training or prediction
pipeline_args.args['time_steps'] = '15' #1 for dense
network_args.network["model_type"] = 'lstm'

if network_args.network["model_type"] == 'conv2d' or 'convlstm':
    pipeline_args.args['expand_dims'] = True



x_t, y_t, x_val, y_val, x_test_t, y_test_t, size = pipeline()

if pipeline_args.args['mode'] == 'training':
    model_train.train_model()
else:
    y_pred = model_predict.predict('0.9944_1.5356_50.9710.h5')
    #if you want pretty graphs and some backtests:
    if pipeline_args.args['expand_dims'] == False:
        y_pred = y_pred[:, -1, :]
    y_pred_mean = remove_mean(y_pred)

    ic_coef(y_test_t, y_pred)
    plot_results_v2(y_test_t, y_pred, no_mean=True)
    vectorized_backtest(y_test_t, y_pred_mean)

