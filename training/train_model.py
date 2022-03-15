import os

import tensorflow as tf
from dotenv import load_dotenv
# from tensorflow.keras.callbacks import ModelCheckpoint
from pipeline.pipelineargs import PipelineArgs
from Data_Processing.data_trim import trim_dataset
# from Networks.structures.Conv1DLSTM import create_convlstm_model
# from Old_and_crap.Ensemble_model import create_model_ensembly
from pipeline.pipeline_structure import pipeline
# from training.callbacks import ResetStatesOnEpochEnd
from Networks.structures.Dense import create_dense_model
from Networks.structures.LSTM_self_att import create_lstm_model
from Networks.structures.Conv1D import create_model
from Networks.structures.Conv1dLSTM_2dconv import create_convlstm_model

load_dotenv()

'''Module for training new models'''
pipeline_args = PipelineArgs.get_instance()

batch_size = pipeline_args.args['batch_size']
time_steps = pipeline_args.args['time_steps']

x_t, y_t, x_val, y_val, x_test_t, y_test_t,size = pipeline(pipeline_args)

def train_model():

    # early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode='min', patience=100)
    #
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.85,
    #                                                  patience=8, min_lr=0.000000000001,
    #                                                  verbose=1, mode='min')
    # reset_states = ResetStatesOnEpochEnd()
    # mcp = ModelCheckpoint(
    #     os.path.join(f'F:\MM\models\\bnbusdt\\1min\\',
    #                  "{val_loss:.8f}_{val_metric_signs:.8f}-best_model-{epoch:02d}.h5"),
    #     monitor='val_loss', verbose=3,
    #     save_best_only=False, save_weights_only=False, mode='min', period=1)

    lstm_model = create_dense_model()


    history_lstm = lstm_model.fit(x=trim_dataset(x_t, batch_size),y=trim_dataset(y_t,batch_size), epochs=10000,
                                  verbose=1, batch_size=batch_size,
                                  shuffle=False, validation_data=(trim_dataset(x_val, batch_size),
                                                                  trim_dataset(y_val, batch_size)),
                                  callbacks=[])


train_model()

