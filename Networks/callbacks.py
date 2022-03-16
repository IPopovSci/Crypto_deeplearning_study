import os
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from Networks.network_config import NetworkParams
from pipeline.pipelineargs import PipelineArgs
from dotenv import load_dotenv
from utility import hash_folder_create

load_dotenv()

network_args = NetworkParams.get_instance()

pipeline_args = PipelineArgs.get_instance()

def callbacks():

    early_stop = EarlyStopping(monitor=network_args.callbacks['monitor'], mode=network_args.callbacks['mode'], patience=network_args.callbacks['es_patience'])

    reduce_lr = ReduceLROnPlateau(monitor=network_args.callbacks['monitor'], factor=network_args.callbacks['rlr_factor'],
                                  patience=network_args.callbacks['rlr_patience'], min_lr=0.00000001,
                                  verbose=1, mode=network_args.callbacks['mode'])

    #hash_folder_create()

    mcp = ModelCheckpoint(
        os.path.join(os.getenv('model_path') + f'\{pipeline_args.args["interval"]}\{pipeline_args.args["ticker"]}\{network_args.network["model_type"]}',
                     "{val_metric_signs_close:.1f}_{val_loss:.8f}_{epoch:02d}.h5"),
        monitor=network_args.callbacks['monitor'], verbose=3,
        save_best_only=True, save_weights_only=False, mode=network_args.callbacks['mode'], period=1)

    return [early_stop,reduce_lr,mcp]