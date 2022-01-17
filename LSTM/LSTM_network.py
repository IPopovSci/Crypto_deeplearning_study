from tensorflow.keras.layers import LSTM, Dense, Input,TimeDistributed
from Arguments import args
from LSTM.callbacks import mean_squared_error_custom
import tensorflow as tf
from keras_self_attention import SeqSelfAttention




def create_lstm_model(x_t):
    BATCH_SIZE = args['batch_size']
    TIME_STEPS = args['time_steps']
    n_components = args['n_components']

    input = Input(batch_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]))
    regularizer = tf.keras.regularizers.l1_l2(1e-4)

# #This is First side-chain: input>LSTM(stateful)>LSTM(stateful)>TD Dense layer. The output is a 3d vector
    LSTM_1 = LSTM(int(80), return_sequences=True, stateful=True,activation='softsign')(input)
#
    LSTM_2 = LSTM(int(40), return_sequences=True, stateful=True,activation='softsign')(LSTM_1)

    Dense_1 = TimeDistributed(Dense(40,activation='softsign'))(LSTM_2)
#This is the attention side-chain: LSTM(Stateless)>LSTM>Attention. The output is a 3d vector
    LSTM_3 = LSTM(int(80), return_sequences=True, stateful=False,activation='softsign')(input)

    LSTM_4 = LSTM(int(40), return_sequences=True, stateful=False,activation='softsign')(LSTM_3)

    attention = SeqSelfAttention(attention_activation='softsign')(LSTM_4)
#Concat the sidechains and provide output (5 values, 2d vector)

    concat = tf.keras.layers.concatenate([Dense_1,attention])

    LSTM_fin = LSTM(200,return_sequences=True,stateful=True,activation='softsign')(concat)

    LSTM_fin_2 = LSTM(150,return_sequences=False,stateful=False,activation='softsign')(LSTM_fin)


    Dense_fin = Dense(200,activation='softsign')(LSTM_fin_2)
#




    output = tf.keras.layers.Dense(5,activation='softsign')(Dense_fin)


    lstm_model = tf.keras.Model(inputs=input, outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    lstm_model.compile(loss=[mean_squared_error_custom,mean_squared_error_custom,mean_squared_error_custom,mean_squared_error_custom,mean_squared_error_custom], optimizer=optimizer)
    return lstm_model
