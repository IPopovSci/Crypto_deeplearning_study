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

    LSTM_1 = LSTM(int(n_components * 0.8), return_sequences=True, stateful=True,dropout=0.3,recurrent_dropout=0.4,kernel_regularizer=regularizer,activation='selu')(input)

    LSTM_2 = LSTM(int(n_components * 0.8 ** 2), return_sequences=True, stateful=True,dropout=0.3,recurrent_dropout=0.4,kernel_regularizer=regularizer,activation='selu')(LSTM_1)

    attention_1 = Attention(int(n_components * 0.8 ** 2))(LSTM_2)

    Dense_1 = tf.keras.layers.Dense(n_components * 0.8 ** 3, activation='selu')(attention_1)

    output = tf.keras.layers.Dense(1, activation='sigmoid')(Dense_1)

    lstm_model = tf.keras.Model(inputs=input, outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(args['LR'], 10000,
                                                            decay_rate=0.95))
    lstm_model.compile(loss=custom_loss, optimizer=optimizer)
    return lstm_model
