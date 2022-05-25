import os

from dotenv import load_dotenv
from keras_self_attention import SeqSelfAttention
from tensorflow.keras.models import load_model

from Data_Processing.data_trim import trim_dataset
from Networks.callbacks import callbacks
from Networks.custom_activation import p_swish,p_softsign
from Networks.losses_metrics import ohlcv_combined, metric_signs_close, ohlcv_cosine_similarity, ohlcv_mse, \
    assymetric_loss, assymetric_combined, metric_loss, metric_profit_ratio, profit_ratio_mse, profit_ratio_cosine, \
    profit_ratio_assymetric
from Networks.network_config import NetworkParams
from Networks.structures.index import create_model
from pipeline.pipelineargs import PipelineArgs
from keras import backend as K

load_dotenv()

'''Module for training new models'''

pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()

batch_size = pipeline_args.args['batch_size']
time_steps = pipeline_args.args['time_steps']

'''Function to train new models
Creates a model based on model_type parameter in the network settings dict.
Performs fitting of x data on the model using the .fit method.'''


def train_model(x_t, y_t, x_val, y_val, model_type=network_args.network["model_type"]):
    model = create_model(model_type)

    history = model.fit(x=trim_dataset(x_t, pipeline_args.args['batch_size']), y=trim_dataset(y_t, pipeline_args.args['batch_size']), epochs=300000,
                        verbose=1, batch_size=pipeline_args.args['batch_size'],
                        shuffle=False, validation_data=(trim_dataset(x_val, pipeline_args.args['batch_size']),
                                                        trim_dataset(y_val, pipeline_args.args['batch_size'])),
                        callbacks=callbacks())


'''Function to continue train models
loads model based on model_name (Must include .h5 at the end!).
Performs fitting of x data on the model using the .fit method.'''
def continue_training(x_t, y_t, x_val, y_val, model_name='Default'):
    filepath = os.getenv('model_path') + f'/{pipeline_args.args["interval"]}/{pipeline_args.args["ticker"]}/{network_args.network["model_type"]}/' + model_name
    saved_model = load_model(filepath=filepath,
                             custom_objects={'metric_loss': metric_loss, 'assymetric_combined': assymetric_combined,
                                             'assymetric_loss': assymetric_loss,
                                             'metric_signs_close': metric_signs_close,
                                             'SeqSelfAttention': SeqSelfAttention, 'ohlcv_combined': ohlcv_combined,
                                             'ohlcv_cosine_similarity': ohlcv_cosine_similarity,
                                             'ohlcv_mse': ohlcv_mse,
                                             'metric_profit_ratio': metric_profit_ratio,
                                             'profit_ratio_mse': profit_ratio_mse,
                                             'profit_ratio_cosine': profit_ratio_cosine,
                                             'profit_ratio_assymetric': profit_ratio_assymetric,
                                             'p_swish': p_swish,
                                             'p_softsign':p_softsign})

    K.set_value(saved_model.optimizer.learning_rate, 0.01) #Use this if you want to set a new learning rate for the model

    history = saved_model.fit(x=trim_dataset(x_t, pipeline_args.args['batch_size']), y=trim_dataset(y_t, pipeline_args.args['batch_size']), epochs=30000,
                              verbose=1, batch_size=pipeline_args.args['batch_size'],
                              shuffle=False, validation_data=(trim_dataset(x_val, pipeline_args.args['batch_size']),
                                                              trim_dataset(y_val, pipeline_args.args['batch_size'])),
                              callbacks=callbacks())
