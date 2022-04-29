import sys
import numpy as np
np.random.seed(1337)
from tensorflow.keras.models import load_model
import os
from keras_self_attention import SeqSelfAttention
from pipeline.pipelineargs import PipelineArgs
from dotenv import load_dotenv
from Networks.network_config import NetworkParams
from Networks.losses_metrics import ohlcv_combined, metric_signs_close, ohlcv_cosine_similarity, ohlcv_mse, \
    assymetric_loss, assymetric_combined, metric_loss, metric_profit_ratio, profit_ratio_mse, profit_ratio_cosine, \
    profit_ratio_assymetric, assymetric_loss_mse
from Networks.custom_activation import cyclemoid
from Networks.custom_activation import p_swish, p_softsign
from keras.layers import Activation
from pathlib import Path

load_dotenv()

pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()
custom_objects = {'metric_loss': metric_loss, 'assymetric_combined': assymetric_combined,
                  'assymetric_loss': assymetric_loss,
                  'metric_signs_close': metric_signs_close,
                  'SeqSelfAttention': SeqSelfAttention, 'ohlcv_combined': ohlcv_combined,
                  'ohlcv_cosine_similarity': ohlcv_cosine_similarity,
                  'ohlcv_mse': ohlcv_mse,
                  'metric_profit_ratio': metric_profit_ratio,
                  'profit_ratio_mse': profit_ratio_mse,
                  'profit_ratio_cosine': profit_ratio_cosine,
                  'profit_ratio_assymetric': profit_ratio_assymetric,
                  'cyclemoid': cyclemoid,
                  'Activation': Activation,
                  'p_swish': p_swish,
                  'p_softsign': p_softsign,
                  'assymetric_loss_mse': assymetric_loss_mse}

'''Does a complete parse of the models folder to fill sql database initially
Returns: Param dictionary in the format model:{parameters}'''
def initial_folder_parse():
    params = {}

    p = Path(sys.path[0])
    filepath = p.parent.joinpath('models')
    # print(filepath)

    for root, dirs, files in os.walk(filepath, topdown=True):

        for name in files:
            if name.endswith('.h5'):
                # print(os.path.join(root, name))
                model_name = name
                #Converting this to pathlib results in some sort of problem with model load
                final_folder = root.split('\\')[-1]

                interval = root.split('\\')[-2]
                ticker = root.split('\\')[-3]

                valid_models_types = ['conv1d', 'conv2d', 'convlstm', 'dense', 'lstm']
                ensemble_models_types = ['ensembly_average']

                if final_folder in valid_models_types:
                    type = final_folder

                    model_full_path = Path(filepath, ticker, interval, final_folder, model_name)
                    # print(os.path.abspath(model_full_path))
                    # print(os.listdir(model_full_path))
                    saved_model = load_model(filepath=model_full_path, custom_objects=custom_objects)
                    config = saved_model.get_config()

                    # print(config['layers'][1])

                    depth = len(config['layers'])

                    input_shape = config['layers'][0]['config']['batch_input_shape']

                    optimizer = saved_model.optimizer.get_config()
                    optimizer_type = optimizer['name']
                    learning_rate = optimizer['learning_rate']
                    b1 = optimizer['beta_1']
                    b2 = optimizer['beta_2']
                    epsilon = optimizer['epsilon']
                    amsgrad = optimizer['amsgrad']
                    decay = optimizer['decay']

                    layer_config = config['layers']

                    for ensemble_type in ensemble_models_types:  # Ensemble models must always exist in one of the folder types above
                        path_ensemble = Path(filepath, ticker, interval, ensemble_type)
                        if model_name in os.listdir(path_ensemble):
                            # print(model_name,'is in',ensemble_type)
                            ensemble = True
                            ensemble_type = ensemble_type
                        else:
                            # print(model_name, 'is not in', ensemble_type)
                            ensemble = False
                            ensemble_type = None

                # params{model_name:{'type':,'depth':,'input_shape':,'optimizer_type':,'optimizer_id':,'ensemble:','lc_id':}} -- this is what our dict will look like
                params[model_name] = {'type': type, 'depth': depth, 'input_shape': input_shape,
                                      'optimizer_type': optimizer_type, 'optimizer_id': None, 'ensemble': ensemble,
                                      'ensemble_type': ensemble_type,
                                      'lc_id': None, 'ticker': ticker, 'interval': interval, 'decay': decay,
                                      'learning_rate': learning_rate, 'b1': b1, 'b2': b2, 'epsilon': epsilon,
                                      'amsgrad': amsgrad, 'lc_config': layer_config}

    return params

'''Parses parameters of a single model
Returns: Dict of parameters'''
def single_model_parse(model_path,ticker,interval):
    #This needs to be ran manually for each new model
    #Till I figure out how to do this automatically with Keras

    saved_model = load_model(filepath=model_path, custom_objects=custom_objects)
    model_name = Path(model_path).name

    config = saved_model.get_config()

    # print(config['layers'][1])

    depth = len(config['layers'])

    input_shape = config['layers'][0]['config']['batch_input_shape']

    optimizer = saved_model.optimizer.get_config()
    optimizer_type = optimizer['name']
    learning_rate = optimizer['learning_rate']
    b1 = optimizer['beta_1']
    b2 = optimizer['beta_2']
    epsilon = optimizer['epsilon']
    amsgrad = optimizer['amsgrad']
    decay = optimizer['decay']

    layer_config = config['layers']

    ensemble = False #Fresh saved model won't be in ensemble by default
    ensemble_type = None

    params = {'model_name':model_name,'type': str(Path(model_path).parts[-2]), 'depth': depth, 'input_shape': input_shape,
                          'optimizer_type': optimizer_type, 'optimizer_id': None, 'ensemble': ensemble,
                          'ensemble_type': ensemble_type,
                          'lc_id': None, 'ticker': ticker, 'interval': interval, 'decay': decay,
                          'learning_rate': learning_rate, 'b1': b1, 'b2': b2, 'epsilon': epsilon,
                          'amsgrad': amsgrad, 'lc_config': layer_config}

    return params
