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
    assymetric_loss, assymetric_combined, metric_loss,metric_profit_ratio,profit_ratio_mse, profit_ratio_cosine,profit_ratio_assymetric
from Backtesting.Backtesting import correct_signs, ic_coef
from plotting import plot_results_v2, plot_ic
from utility import remove_mean,remove_std
from Backtesting.Backtesting import vectorized_backtest

load_dotenv()

'''Module for training new models'''
pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()

batch_size = pipeline_args.args['batch_size']
time_steps = pipeline_args.args['time_steps']


'''Function to predict using existing model
Accepts: string model filename.
Returns: numpy array with predictions.'''


def predict(x_test_t, y_test_t,model_name='Default'):
    saved_model = load_model(filepath=(os.getenv(
        'model_path') + f'/{pipeline_args.args["interval"]}/{pipeline_args.args["ticker"]}/{network_args.network["model_type"]}/' + model_name),
                             custom_objects={'metric_loss': metric_loss, 'assymetric_combined': assymetric_combined,
                                             'assymetric_loss': assymetric_loss,
                                             'metric_signs_close': metric_signs_close,
                                             'SeqSelfAttention': SeqSelfAttention, 'ohlcv_combined': ohlcv_combined,
                                             'ohlcv_cosine_similarity': ohlcv_cosine_similarity,
                                             'ohlcv_mse': ohlcv_mse,
                                             'metric_profit_ratio': metric_profit_ratio,
                                             'profit_ratio_mse': profit_ratio_mse,
                                             'profit_ratio_cosine':profit_ratio_cosine,
                                             'profit_ratio_assymetric':profit_ratio_assymetric})
    saved_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00000005),
                        loss=ohlcv_combined, metrics=metric_signs_close)


    y_pred = saved_model.predict(trim_dataset(x_test_t[:], batch_size), batch_size=batch_size)

    saved_model.summary()

    return y_pred
