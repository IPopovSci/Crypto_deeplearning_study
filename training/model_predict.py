from pipeline.pipeline_structure import pipeline
from Data_Processing.data_trim import trim_dataset
from tensorflow.keras.models import load_model
import os
from keras_self_attention import SeqSelfAttention
import tensorflow as tf
from pipeline.pipelineargs import PipelineArgs
from dotenv import load_dotenv
from Networks.network_config import NetworkParams
from Networks.losses_metrics import ohlcv_combined,metric_signs_close,ohlcv_cosine_similarity,ohlcv_mse
from utility import unscale
from Backtesting.Backtesting import correct_signs
from plotting import plot_results
from utility import remove_mean
import numpy as np

load_dotenv()

'''Module for training new models'''
pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()

batch_size = pipeline_args.args['batch_size']
time_steps = pipeline_args.args['time_steps']
pipeline_args.args['mode'] = 'prediction'

x_t, y_t, x_val, y_val, x_test_t, y_test_t,size = pipeline(pipeline_args)
#TODO: ticker name of data is different from API - mini facade? During training data split destroys testing data, need to disable it
def predict(model_name='Default'):
    print(os.getenv('model_path') + f'\{pipeline_args.args["interval"]}\{pipeline_args.args["ticker"]}\{network_args.network["model_type"]}\\'+ f'{model_name}.h5')
    saved_model = load_model(filepath=(os.getenv('model_path') + f'\{pipeline_args.args["interval"]}\{pipeline_args.args["ticker"]}\{network_args.network["model_type"]}\\'+ f'{model_name}.h5'),
                             custom_objects={'metric_signs_close':metric_signs_close,'SeqSelfAttention': SeqSelfAttention,'ohlcv_combined':ohlcv_combined,'ohlcv_cosine_similarity':ohlcv_cosine_similarity,'ohlcv_mse':ohlcv_mse})
    saved_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00000005),
                        loss= ohlcv_combined,metrics=metric_signs_close)


    y_pred = saved_model.predict(trim_dataset(x_test_t, batch_size), batch_size=batch_size)

    y_pred = y_pred[:, -1, :]  # Because Dense predictions will have timesteps

    #y_true, y_pred = unscale(y_test_t,y_pred)


    y_pred_nomean = remove_mean(y_pred)

    y_true = y_test_t[:,3]
    y_pred = y_pred[:,3]
    y_pred_nomean = y_pred_nomean[:,3]

    window_size = 50
    window_true = np.ones(int(window_size*3)) / float(window_size*3)
    window_pred = np.ones(int(window_size)) / float(window_size)
    y_true_average = np.convolve(y_true, window_true, 'same')
    y_pred_average = np.convolve(y_pred, window_pred, 'same')

    plot_results(y_pred_average,y_true_average)

    print(correct_signs(y_true,y_pred))
    print(correct_signs(y_true,y_pred_nomean))

predict('0.00388755_02_51.6')
