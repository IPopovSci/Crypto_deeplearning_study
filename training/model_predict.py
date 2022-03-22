from pipeline.pipeline_structure import pipeline
from Data_Processing.data_trim import trim_dataset
from tensorflow.keras.models import load_model
import os
from keras_self_attention import SeqSelfAttention
import tensorflow as tf
from pipeline.pipelineargs import PipelineArgs
from dotenv import load_dotenv
from Networks.network_config import NetworkParams
from Networks.losses_metrics import ohlcv_combined, metric_signs_close, ohlcv_cosine_similarity, ohlcv_mse, \
    assymetric_loss, assymetric_combined, metric_loss
from Backtesting.Backtesting import correct_signs, ic_coef
from plotting import plot_results_v2, plot_ic
from Backtesting.pyfolio import pyfolio_rolling_returns
from utility import remove_mean,remove_std

load_dotenv()

'''Module for training new models'''
pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()

batch_size = pipeline_args.args['batch_size']
time_steps = pipeline_args.args['time_steps']
pipeline_args.args['mode'] = 'prediction'

x_t, y_t, x_val, y_val, x_test_t, y_test_t, size = pipeline()

'''Function to predict using existing model
Accepts: string model filename.
Returns: numpy array with predictions.'''


def predict(model_name='Default'):
    saved_model = load_model(filepath=(os.getenv(
        'model_path') + f'\{pipeline_args.args["interval"]}\{pipeline_args.args["ticker"]}\{network_args.network["model_type"]}\\' + model_name),
                             custom_objects={'metric_loss': metric_loss, 'assymetric_combined': assymetric_combined,
                                             'assymetric_loss': assymetric_loss,
                                             'metric_signs_close': metric_signs_close,
                                             'SeqSelfAttention': SeqSelfAttention, 'ohlcv_combined': ohlcv_combined,
                                             'ohlcv_cosine_similarity': ohlcv_cosine_similarity,
                                             'ohlcv_mse': ohlcv_mse})
    saved_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00000005),
                        loss=ohlcv_combined, metrics=metric_signs_close)

    y_pred = saved_model.predict(trim_dataset(x_test_t[:], batch_size), batch_size=batch_size)

    y_pred = saved_model.predict(trim_dataset(x_test_t[:], batch_size), batch_size=batch_size)

    return y_pred


y_pred = predict('0.9559_6.7297_53.0357.h5')

if pipeline_args.args['expand_dims'] == False:
    y_pred = y_pred[:, -1, :]

y_pred_mean = remove_mean(y_pred)

ic_coef(y_test_t, y_pred)
correct_signs(y_test_t, y_pred)

plot_results_v2(y_test_t, y_pred, no_mean=True)

pyfolio_rolling_returns(y_test_t[:, 2], y_pred_mean[:, 2])

