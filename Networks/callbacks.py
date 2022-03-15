import os
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from Networks.network_config import NetworkParams
from pipeline.pipelineargs import PipelineArgs
from dotenv import load_dotenv

load_dotenv()

clb_args = NetworkParams.get_instance()

pipeline_args = PipelineArgs.get_instance()

def callbacks():

    early_stop = EarlyStopping(monitor=clb_args.callbacks['monitor'],mode=clb_args.callbacks['mode'], patience=clb_args.callbacks['es_patience'])

    reduce_lr = ReduceLROnPlateau(monitor=clb_args.callbacks['monitor'], factor=clb_args.callbacks['rlr_factor'],
                                                     patience=clb_args.callbacks['rlr_patience'], min_lr=0.000000001,
                                                     verbose=1, mode=clb_args.callbacks['mode'])
    mcp = ModelCheckpoint(
        os.path.join(os.getenv('model_path') + f'\{pipeline_args.args["interval"]}\{pipeline_args.args["ticker"]}\{clb_args.network["model_type"]}',
                     "\{val_loss:.8f}_{epoch:02d}.h5"),
        monitor=clb_args.callbacks['monitor'], verbose=3,
        save_best_only=True, save_weights_only=False, mode=clb_args.callbacks['mode'], period=1)

    return [early_stop,reduce_lr,mcp]