from pipeline import data_prep
from Arguments import args
from Data_Processing.data_trim import trim_dataset
from LSTM.callbacks import mean_squared_error_custom,custom_cosine_similarity,metric_signs,custom_mean_absolute_error,stock_loss
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import os
from keras_self_attention import SeqSelfAttention
import tensorflow as tf
from Backtesting.Backtesting import correct_signs
import joblib
import tensorflow.keras.backend as K

BATCH_SIZE = args['batch_size']
ticker = 'bnbusdt'
def predict(model_name='Default',update=False):
    x_t, y_t, x_val, y_val, x_test_t, y_test_t,size = data_prep('pancake',initial_training=False,batch=True,SS_path = 'F:\MM\scalers\\bnbusdt_ss_pancake1min',MM_path = 'F:\MM\scalers\\bnbusdt_mm_pancake1min',big_update=False)

    saved_model = load_model(os.path.join(f'F:\MM\production\pancake_predictions\models\\1min\\', f'{model_name}.h5'),
                             custom_objects={'stock_loss':stock_loss,'custom_mean_absolute_error':custom_mean_absolute_error,'metric_signs':metric_signs,'SeqSelfAttention': SeqSelfAttention,'custom_cosine_similarity':custom_cosine_similarity,'mean_squared_error_custom':mean_squared_error_custom})
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

    y_pred_lstm = saved_model.predict(trim_dataset(x_test_t[-800:-600], BATCH_SIZE), batch_size=BATCH_SIZE)

    y_pred_lstm = saved_model.predict(trim_dataset(x_test_t[-600:], BATCH_SIZE), batch_size=BATCH_SIZE)

    MM_path = 'F:\MM\scalers\\bnbusdt_mm_pancake1min'
    SS_path = 'F:\MM\scalers\\bnbusdt_ss_pancake1min'

    mm_y = joblib.load(MM_path + ".y")
    sc_y = joblib.load(SS_path + ".y")

    y_test_t = (((y_test_t - K.constant(mm_y.min_)) / K.constant(mm_y.scale_))* sc_y.scale_) + sc_y.mean_


    y_pred_lstm = (((y_pred_lstm - K.constant(mm_y.min_)) / K.constant(mm_y.scale_)) * sc_y.scale_) + sc_y.mean_



    print(correct_signs(y_test_t[-600:],y_pred_lstm[-600:]))
    print(y_test_t[-150:])
    print(y_pred_lstm[-150:])


predict('569.07647705_54.41176605-best_model-01',False)
