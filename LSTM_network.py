from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from Arguments import args
from callbacks import custom_loss,custom_loss_hinge, stock_loss_money,stock_loss
import tensorflow as tf
from attention import Attention


def create_lstm_model(x_t):
    BATCH_SIZE = args['batch_size']
    TIME_STEPS = args['time_steps']
    n_components = args['n_components']

    input = Input(batch_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]))
    regularizer = tf.keras.regularizers.l1_l2(1e-3)


    LSTM_1 = LSTM(int(64), return_sequences=True, stateful=True, kernel_regularizer=regularizer,
             recurrent_dropout=0.5, dropout=0.5, bias_regularizer=tf.keras.regularizers.l2(1e-4),
             activity_regularizer=tf.keras.regularizers.l2(1e-5))(input)

    LSTM_2 = LSTM(int(32), return_sequences=True, stateful=False, kernel_regularizer=regularizer,
                  recurrent_dropout=0.5, dropout=0.5, bias_regularizer=tf.keras.regularizers.l2(1e-4),
                  activity_regularizer=tf.keras.regularizers.l2(1e-5))(LSTM_1)

    attention_1 = Attention(int(32))(LSTM_2)



    Dense_2 = tf.keras.layers.Dense(int(16), activation='relu')(attention_1)

    Dense_3 = tf.keras.layers.Dense(int(8), activation='relu')(Dense_2)

    output = tf.keras.layers.Dense(1, activation='relu')(Dense_3)

    lstm_model = tf.keras.Model(inputs=input, outputs=output)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(args['LR'], 2500,
                                                                                                     # decay_rate=0.95),clipnorm=1.,clipvalue=1.)
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=1.,rho= 0.9)
    lstm_model.compile(loss=stock_loss_money, optimizer=optimizer)
    return lstm_model
