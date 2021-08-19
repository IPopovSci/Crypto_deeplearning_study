from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from Arguments import args
from callbacks import custom_loss,custom_loss_hinge, stock_loss_money
import tensorflow as tf
from attention import Attention


def create_lstm_model(x_t):
    BATCH_SIZE = args['batch_size']
    TIME_STEPS = args['time_steps']
    n_components = args['n_components']
    regularizer = tf.keras.regularizers.l1_l2(l1=0, l2=0)

    input = Input(batch_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]))


    LSTM_1 = LSTM(int(n_components*3), return_sequences=True, stateful=True, kernel_regularizer=regularizer,
             recurrent_dropout=0.3, dropout=0.3,
             activity_regularizer=tf.keras.regularizers.l2(1e-6))(input)

    LSTM_2 = LSTM(int(n_components*2), return_sequences=True, stateful=False, kernel_regularizer=regularizer,
             dropout=0.3, recurrent_dropout=0.3,
             activity_regularizer=tf.keras.regularizers.l2(1e-6))(LSTM_1)


    attention_1 = Attention(int(n_components*1.5))(LSTM_2)


    Dense_1 = tf.keras.layers.Dense(int(n_components))(attention_1)

    output = tf.keras.layers.Dense(1)(Dense_1)

    lstm_model = tf.keras.Model(inputs=input, outputs=output)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(args['LR'], 2500,
    #                                                                                                   decay_rate=0.95))
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=1.)
    lstm_model.compile(loss=stock_loss_money, optimizer=optimizer)
    return lstm_model
