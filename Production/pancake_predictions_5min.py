from pipeline import data_prep
from Arguments import args
from Data_Processing.data_trim import trim_dataset
from LSTM.callbacks import mean_squared_error_custom,custom_cosine_similarity,metric_signs
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import os
from keras_self_attention import SeqSelfAttention
import tensorflow as tf
from Backtesting.Backtesting import correct_signs

BATCH_SIZE = args['batch_size']
ticker = 'bnbusdt'
def predict(model_name='Default',update=False):
    x_t, y_t, x_val, y_val, x_test_t, y_test_t,size = data_prep('pancake',initial_training=False,batch=True,SS_path = 'F:\MM\scalers\\bnbusdt_ss_pancake5min',MM_path = 'F:\MM\scalers\\bnbusdt_mm_pancake5min',big_update=False)

    saved_model = load_model(os.path.join(f'F:\MM\production\pancake_predictions\models\\5min\\', f'{model_name}.h5'),
                             custom_objects={'metric_signs':metric_signs,'SeqSelfAttention': SeqSelfAttention,'custom_cosine_similarity':custom_cosine_similarity,'mean_squared_error_custom':mean_squared_error_custom})
    saved_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                        loss= custom_cosine_similarity,metrics=metric_signs)

    mcp = ModelCheckpoint(
        os.path.join(f'F:\MM\production\pancake_predictions\models\\5min\\',
                     "{metric_signs:.8f}_{loss:.8f}-best_model-{epoch:02d}.h5"),
        monitor='val_loss', verbose=3,
        save_best_only=False, save_weights_only=False, mode='min', period=1)

    if update == True:
        history_lstm = saved_model.fit(trim_dataset(x_test_t[-1000:-200], BATCH_SIZE), trim_dataset(y_test_t[-1000:-200], BATCH_SIZE),
                                       epochs=1, verbose=1, batch_size=BATCH_SIZE,
                                       shuffle=False,callbacks=[mcp])
        saved_model.reset_states()

    #y_pred_lstm = saved_model.predict(trim_dataset(x_test_t[:-200], BATCH_SIZE), batch_size=BATCH_SIZE)

    y_pred_lstm = saved_model.predict(trim_dataset(x_test_t[-150:], BATCH_SIZE), batch_size=BATCH_SIZE)

    print(correct_signs(y_test_t[-10:],y_pred_lstm[-10:]))
    print(y_test_t[-10:])
    print(y_pred_lstm[-1])


predict('-0.72404236_92.19999695-best_model-01',False)
