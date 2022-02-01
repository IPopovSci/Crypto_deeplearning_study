from pipeline import data_prep_batch_2, data_prep
from Arguments import args
from Data_Processing.data_trim import trim_dataset
from tensorflow.keras.callbacks import ModelCheckpoint
# from plotting import plot_results
import os
import tensorflow as tf
from LSTM.LSTM_network import create_lstm_model as create_model

BATCH_SIZE = args['batch_size']

ticker = args['ticker']

def train_model_batch(start,increment,model_name='Default'):
    x_t, y_t, x_val, y_val, x_test_t, y_test_t,size = data_prep('CSV',initial_training=True,batch=True)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_metric_signs', factor=0.5,
                                                     patience=4, min_lr=0.00000001,
                                                     verbose=1, mode='max')
    mcp = ModelCheckpoint(
        os.path.join(f'F:\MM\models\{ticker}\\',
                     "{val_loss:.8f}_{loss:.8f}-best_model-{epoch:02d}.h5"),
        monitor='val_loss', verbose=3,
        save_best_only=False, save_weights_only=False, mode='min', period=1)

    x_ts, y_ts = data_prep_batch_2(x_t, y_t, start, increment)
    lstm_model = create_model(x_ts) #We can eliminate dependancy on x_ts (And therefore previous line) if we get the number of features somewhere in the pipeline
    end = increment + start

    while end < size:
        x_train, y_train = data_prep_batch_2(x_t,y_t,start, end)
        history_lstm = lstm_model.fit(trim_dataset(x_train, BATCH_SIZE), trim_dataset(y_train, BATCH_SIZE), epochs=1,
                                      verbose=1, batch_size=BATCH_SIZE,
                                      shuffle=False, validation_data=(trim_dataset(x_val, BATCH_SIZE),
                                                                      trim_dataset(y_val, BATCH_SIZE)),
                                      callbacks=[mcp,reduce_lr])
        start = end
        end += increment
        lstm_model.reset_states()
        if end > size:
            end = increment
            start = 0

train_model_batch(0,25000, ticker)
