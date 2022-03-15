from pipeline.pipeline_structure import pipeline
from Data_Processing.data_trim import trim_dataset
from tensorflow.keras.models import load_model
import os
from keras_self_attention import SeqSelfAttention
import tensorflow as tf
from Backtesting.Backtesting import correct_signs
from pipeline.pipelineargs import PipelineArgs

BATCH_SIZE = args['batch_size']
ticker = 'bnbusdt'
pipeline_args = PipelineArgs.get_instance()

def predict(model_name='Default',update=False):
    x_t, y_t, x_val, y_val, x_test_t, y_test_t,size = pipeline('CSV')

    saved_model = load_model(os.path.join(f'F:\MM\models\\bnbusdt\\', f'{model_name}.h5'),
                             custom_objects={'metric_signs':metric_signs,'SeqSelfAttention': SeqSelfAttention,'custom_cosine_similarity':custom_cosine_similarity,'mean_squared_error_custom':mean_squared_error_custom})
    saved_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00000005),
                        loss= custom_cosine_similarity,metrics=metric_signs)
    saved_model.reset_states()

    if update == True:
        history_lstm = saved_model.fit(trim_dataset(x_test_t[-1000:-500], BATCH_SIZE), trim_dataset(y_test_t[-1000:-500], BATCH_SIZE),
                                       epochs=1, verbose=1, batch_size=BATCH_SIZE,
                                       shuffle=False)


    y_pred_lstm = saved_model.predict(trim_dataset(x_test_t[-500:], BATCH_SIZE), batch_size=BATCH_SIZE)
    print(correct_signs(y_test_t[-500:],y_pred_lstm))


predict(ticker,'0.12958081_59.17741776-best_model-01',False)

def train_model():


    lstm_model = create_model()


    history_lstm = lstm_model.fit(x=trim_dataset(x_t, batch_size),y=trim_dataset(y_t,batch_size), epochs=10000,
                                  verbose=1, batch_size=batch_size,
                                  shuffle=False, validation_data=(trim_dataset(x_val, batch_size),
                                                                  trim_dataset(y_val, batch_size)),
                                  callbacks=callbacks())


train_model()

