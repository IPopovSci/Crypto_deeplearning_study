from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Input
from keras import optimizers
from Arguments import args
from build_timeseries import build_timeseries
from callbacks import custom_loss
from keras.layers import TimeDistributed
import tensorflow as tf

def create_lstm_model(x_t):
    BATCH_SIZE = args['batch_size']
    TIME_STEPS = args['time_steps']
    n_components = args['n_components']

    input = Input(batch_shape=(BATCH_SIZE,TIME_STEPS,x_t.shape[2]))

    #Layer_norm_0 = tf.keras.layers.LayerNormalization(center=False,scale=False)(input)

    LSTM_1 = LSTM(int(n_components * 0.8),return_sequences=True,stateful=True)(input)

    #Layer_norm_1 = tf.keras.layers.LayerNormalization(center=False,scale=False)(LSTM_1)

    LSTM_2 = LSTM(int(n_components * 0.8**2),return_sequences=True,stateful=True)(LSTM_1)

    #Layer_norm_1 = tf.keras.layers.LayerNormalization(center=False,scale=False)(LSTM_2)


    Dense_1 = tf.keras.layers.Dense(n_components * 0.8**3,activation='relu')(LSTM_2)

    #Layer_norm_1 = tf.keras.layers.LayerNormalization(center=False,scale=False)(Dense_1)



    output = tf.keras.layers.Dense(1,activation='sigmoid')(Dense_1)

    lstm_model = tf.keras.Model(inputs = input, outputs = output)
    optimizer = optimizers.Adam(lr=args["LR"])
    lstm_model.compile(loss=custom_loss, optimizer=optimizer)





    # lstm_model = Sequential()
    # lstm_model.add(LSTM(30, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]),
    #                     dropout=0.05, recurrent_dropout=0.05,
    #                     stateful=True, return_sequences=True,
    #                     kernel_initializer='random_uniform'))
    # lstm_model.add(Dropout(0.4))
    #
    # lstm_model.add(LSTM(20, dropout=0.05,stateful=True, recurrent_dropout = 0.05,return_sequences=False))
    # lstm_model.add(Dropout(0.4))
    #
    # lstm_model.add(Dense(10, activation='relu'))
    # lstm_model.add(Dropout(0.4))
    # lstm_model.add(Dense(1, activation='sigmoid'))
    #
    # # compile the model
    # optimizer = optimizers.Adam(lr=args["LR"])
    # lstm_model.compile(loss=custom_loss, optimizer=optimizer)

    return lstm_model