from pipeline import data_prep
from Arguments import args
from Data_Processing.data_trim import trim_dataset
from LSTM.callbacks import mean_squared_error_custom,custom_cosine_similarity,metric_signs,custom_mean_absolute_error
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import os
from keras_self_attention import SeqSelfAttention
import tensorflow as tf
from Backtesting.Backtesting import correct_signs

BATCH_SIZE = args['batch_size']
ticker = 'bnbusdt'
def predict(model_name='Default',update=False):
    x_t, y_t, x_val, y_val, x_test_t, y_test_t,size = data_prep('pancake',initial_training=False,batch=True,SS_path = 'F:\MM\scalers\\bnbusdt_ss_pancake1min',MM_path = 'F:\MM\scalers\\bnbusdt_mm_pancake1min',big_update=False)

    saved_model = load_model(os.path.join(f'F:\MM\production\pancake_predictions\models\\1min\\', f'{model_name}.h5'),
                             custom_objects={'custom_mean_absolute_error':custom_mean_absolute_error,'metric_signs':metric_signs,'SeqSelfAttention': SeqSelfAttention,'custom_cosine_similarity':custom_cosine_similarity,'mean_squared_error_custom':mean_squared_error_custom})
    saved_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005),
                        loss= custom_mean_absolute_error,metrics=metric_signs)

    mcp = ModelCheckpoint(
        os.path.join(f'F:\MM\production\pancake_predictions\models\\1min\\',
                     "{metric_signs:.8f}_{loss:.8f}-best_model-{epoch:02d}.h5"),
        monitor='val_loss', verbose=3,
        save_best_only=False, save_weights_only=False, mode='min', period=1)

    if update == True:
        history_lstm = saved_model.fit(trim_dataset(x_test_t[-900:-200], BATCH_SIZE), trim_dataset(y_test_t[-900:-200], BATCH_SIZE),
                                       epochs=1, verbose=1, batch_size=BATCH_SIZE,
                                       shuffle=False,callbacks=[mcp])
        saved_model.reset_states()

    y_pred_lstm = saved_model.predict(trim_dataset(x_test_t[-800:-200], BATCH_SIZE), batch_size=BATCH_SIZE)

    y_pred_lstm = saved_model.predict(trim_dataset(x_test_t[-200:], BATCH_SIZE), batch_size=BATCH_SIZE)

    print(correct_signs(y_test_t[-100:],y_pred_lstm[-100:]))
    print(y_test_t[-50:])
    print(y_pred_lstm[-50:])


predict('-50.19338989_50.42552185-best_model-01',False)
