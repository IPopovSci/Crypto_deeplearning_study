from dotenv import load_dotenv
from tensorflow.keras.callbacks import ModelCheckpoint
from pipeline.pipelineargs import PipelineArgs
from Data_Processing.data_trim import trim_dataset
from pipeline.pipeline_structure import pipeline
from tensorflow.keras.models import load_model
from Networks.losses_metrics import ohlcv_combined, metric_signs_close, ohlcv_cosine_similarity, ohlcv_mse, \
    assymetric_loss, assymetric_combined, metric_loss,metric_profit_ratio,profit_ratio_mse, profit_ratio_cosine,profit_ratio_assymetric
from Networks.structures._index import create_model
from Networks.callbacks import callbacks
from keras_self_attention import SeqSelfAttention
from Networks.network_config import NetworkParams
import os
import tensorflow as tf

load_dotenv()

'''Module for training new models'''

pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()

batch_size = pipeline_args.args['batch_size']
time_steps = pipeline_args.args['time_steps']


'''Function to train new models
Creates a model based on model_type parameter in the network settings dict.
Performs fitting of x data on the model using the .fit method'''


def train_model(x_t, y_t, x_val, y_val,model_type = network_args.network["model_type"]):
    model = create_model(model_type)

    history = model.fit(x=trim_dataset(x_t, batch_size), y=trim_dataset(y_t, batch_size), epochs=3000,
                                  verbose=1, batch_size=batch_size,
                                  shuffle=False, validation_data=(trim_dataset(x_val, batch_size),
                                                                  trim_dataset(y_val, batch_size)),
                                  callbacks=callbacks())

def continue_training(x_t, y_t, x_val, y_val,model_name='Default'):
    saved_model = load_model(filepath=(os.getenv(
        'model_path') + f'\{pipeline_args.args["interval"]}\{pipeline_args.args["ticker"]}\{network_args.network["model_type"]}\\' + model_name),
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

    # saved_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=network_args.network['lr'],amsgrad = True),
    #                     loss=profit_ratio_assymetric, metrics=[metric_signs_close, ohlcv_cosine_similarity, ohlcv_mse])

    history_lstm = saved_model.fit(x=trim_dataset(x_t, batch_size), y=trim_dataset(y_t, batch_size), epochs=3000,
                                  verbose=1, batch_size=batch_size,
                                  shuffle=False, validation_data=(trim_dataset(x_val, batch_size),
                                                                  trim_dataset(y_val, batch_size)),
                                  callbacks=callbacks())

