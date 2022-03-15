from dotenv import load_dotenv
from tensorflow.keras.callbacks import ModelCheckpoint
from pipeline.pipelineargs import PipelineArgs
from Data_Processing.data_trim import trim_dataset
from pipeline.pipeline_structure import pipeline
from Networks.structures._index import create_model
from Networks.callbacks import callbacks

load_dotenv()

'''Module for training new models'''
pipeline_args = PipelineArgs.get_instance()

batch_size = pipeline_args.args['batch_size']
time_steps = pipeline_args.args['time_steps']

x_t, y_t, x_val, y_val, x_test_t, y_test_t,size = pipeline(pipeline_args)

def train_model():


    lstm_model = create_model()


    history_lstm = lstm_model.fit(x=trim_dataset(x_t, batch_size),y=trim_dataset(y_t,batch_size), epochs=10000,
                                  verbose=1, batch_size=batch_size,
                                  shuffle=False, validation_data=(trim_dataset(x_val, batch_size),
                                                                  trim_dataset(y_val, batch_size)),
                                  callbacks=callbacks())


train_model()

