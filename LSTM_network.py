from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Input
from keras import optimizers
from Arguments import args
from build_timeseries import build_timeseries
from callbacks import custom_loss
from keras.layers import TimeDistributed
import tensorflow as tf
from attention import Attention


def create_lstm_model(x_t):
    BATCH_SIZE = args['batch_size']
    TIME_STEPS = args['time_steps']
    n_components = args['n_components']
    regularizer = tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)

    input = Input(batch_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]))

    LSTM_1 = LSTM(int(n_components), return_sequences=True, stateful=True, kernel_regularizer=regularizer,
             recurrent_dropout=0.5, dropout=0.5, bias_regularizer=tf.keras.regularizers.l2(1e-4),
             activity_regularizer=tf.keras.regularizers.l2(1e-5))(input)

    LSTM_2 = LSTM(int(n_components*0.8), return_sequences=True, stateful=True, kernel_regularizer=regularizer,
             dropout=0.5, recurrent_dropout=0.5,bias_regularizer=tf.keras.regularizers.l2(1e-4),
             activity_regularizer=tf.keras.regularizers.l2(1e-5))(LSTM_1)

    attention_1 = Attention(int(n_components * 0.8))(LSTM_2)

    Dense_1 = tf.keras.layers.Dense(int(n_components * 0.8 ** 2), activation='selu')(attention_1)

    output = tf.keras.layers.Dense(1, activation='sigmoid')(Dense_1)

    lstm_model = tf.keras.Model(inputs=input, outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(args['LR'],200,
                                                                                                      decay_rate=0.90))
    lstm_model.compile(loss=custom_loss, optimizer=optimizer)
    return lstm_model
