import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from Networks.network_config import NetworkParams
from pipeline.pipelineargs import PipelineArgs
from dotenv import load_dotenv
import keras

load_dotenv()

network_args = NetworkParams.get_instance()

pipeline_args = PipelineArgs.get_instance()

'''Function that returns required callbacks for the neural networks related to training'''
class ResetStatesOnEpochEnd(keras.callbacks.Callback):
    def __init__(self):
        super(ResetStatesOnEpochEnd, self).__init__()

    def on_epoch_end(self,epoch,logs=None):
        self.model.reset_states()
        print((self.model.output))
        print('states are reset!')

reset_states = ResetStatesOnEpochEnd()
def callbacks():
    early_stop = EarlyStopping(monitor=network_args.callbacks['monitor'], mode=network_args.callbacks['mode'],
                               patience=network_args.callbacks['es_patience'])

    reduce_lr = ReduceLROnPlateau(monitor=network_args.callbacks['monitor'],
                                  factor=network_args.callbacks['rlr_factor'],
                                  patience=network_args.callbacks['rlr_patience'], min_lr=0.00000001,
                                  verbose=1, mode=network_args.callbacks['mode'])


    filepath = os.getenv(
        'model_path') + f'/{pipeline_args.args["interval"]}/{pipeline_args.args["ticker"]}/{network_args.network["model_type"]}'
    mcp = ModelCheckpoint(
        os.path.join(filepath,
                     "{val_ohlcv_cosine_similarity:.4f}_{val_loss:.4f}_{val_metric_signs_close:.4f}.h5"),
        monitor=network_args.callbacks['monitor'], verbose=3,
        save_best_only=True, save_weights_only=False, mode=network_args.callbacks['mode'], period=1)

    return [early_stop, reduce_lr, mcp,reset_states]


'''A callback that resets the states of stateful network at the end of each epoch'''



