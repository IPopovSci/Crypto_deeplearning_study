from Networks.structures.dense_model import dense_model
from Networks.structures.lstm_att_model import lstm_att_model
from Networks.structures.conv1d_model import conv1d_model
from Networks.structures.conv1d_lstm_model import convlstm_model
from Networks.network_config import  NetworkParams

params = NetworkParams.get_instance()

def create_model(model_type=params.network['model_type']):
    if model_type == 'dense':
        model = dense_model()
    elif model_type == 'lstm':
        model = lstm_att_model()
    elif model_type == 'conv1d':
        model = conv1d_model()
    elif model_type == 'convlstm':
        model = convlstm_model()
    else:
        print('No suitable model. Avaliable models: dense,lstm,conv1d,convlstm')
    return model
