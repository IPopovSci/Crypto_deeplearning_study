from Networks.network_config import NetworkParams
from Networks.structures.conv1d_lstm_model import convlstm_model
from Networks.structures.conv1d_model import conv1d_model
from Networks.structures.conv2d import conv2d_model
from Networks.structures.dense_model import dense_model
from Networks.structures.lstm_model import lstm_att_model

params = NetworkParams.get_instance()

'''Function that creates a model based on model_type parameter.
Accepts: model_type parameter (string).
Returns: keras model object.'''


def create_model(model_type=params.network['model_type']):
    if model_type == 'dense':
        model = dense_model()
    elif model_type == 'lstm':
        model = lstm_att_model()
    elif model_type == 'conv1d':
        model = conv1d_model()
    elif model_type == 'convlstm':
        model = convlstm_model()
    elif model_type == 'conv2d':
        model = conv2d_model()
    else:
        print('No suitable model. Available models: dense, lstm, conv1d, convlstm, conv2d')

    return model


'''Potential future functionality for hash-based model differentiation.'''


def hash_model(model):
    model_json = model.to_json()
    model_hash = hash(model_json)
    params.network['model_hash'] = model_hash
