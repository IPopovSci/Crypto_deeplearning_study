from tensorflow.keras.layers import LSTM, Dense, Input,TimeDistributed
from Arguments import args
from LSTM.callbacks import mean_squared_error_custom,custom_cosine_similarity
import tensorflow as tf
from keras_self_attention import SeqSelfAttention
from tensorflow.keras import initializers



def create_lstm_model(x_t):
    BATCH_SIZE = args['batch_size']
    TIME_STEPS = args['time_steps']
    n_components = args['n_components']

    input = Input(batch_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]))
    regularizer = tf.keras.regularizers.l1_l2(1e-4)
    kernel_init = initializers.RandomNormal(stddev=0.01)

# #This is First side-chain: input>LSTM(stateful)>LSTM(stateful)>TD Dense layer. The output is a 3d vector
    LSTM_1 = LSTM(int(100), return_sequences=True, stateful=True,activation='softsign',kernel_initializer=kernel_init,bias_initializer=kernel_init)(input)
#
    LSTM_2 = LSTM(int(75), return_sequences=True, stateful=True,activation='softsign',kernel_initializer=kernel_init,bias_initializer=kernel_init)(LSTM_1)

    Dense_1 = TimeDistributed(Dense(75,activation='softsign',kernel_initializer=kernel_init,bias_initializer=kernel_init))(LSTM_2)
#This is the attention side-chain: LSTM(Stateless)>LSTM>Attention. The output is a 3d vector
    LSTM_3 = LSTM(int(100), return_sequences=True, stateful=True,activation='softsign',kernel_initializer=kernel_init,bias_initializer=kernel_init)(input)

    LSTM_4 = LSTM(int(75), return_sequences=True, stateful=False,activation='softsign',kernel_initializer=kernel_init,bias_initializer=kernel_init)(LSTM_3)

    attention_1 = SeqSelfAttention(attention_activation='softsign',attention_type='additive',kernel_initializer=kernel_init,bias_initializer=kernel_init)(LSTM_4)
# This is the attention side-chain: LSTM(Stateless)>LSTM>Attention. The output is a 3d vector
    LSTM_5 = LSTM(int(100), return_sequences=True, stateful=True, activation='softsign',kernel_initializer=kernel_init,bias_initializer=kernel_init)(input)

    LSTM_6 = LSTM(int(75), return_sequences=True, stateful=False, activation='softsign',kernel_initializer=kernel_init,bias_initializer=kernel_init)(LSTM_5)

    attention_2 = SeqSelfAttention(attention_activation='softsign',attention_type='multiplicative',kernel_initializer=kernel_init,bias_initializer=kernel_init)(LSTM_6)
#Concat the sidechains and provide output (5 values, 2d vector)

    concat = tf.keras.layers.concatenate([Dense_1,attention_1,attention_2,input])

    LSTM_fin = LSTM(300,return_sequences=True,stateful=True,activation='softsign',kernel_initializer=kernel_init,bias_initializer=kernel_init)(concat)

    LSTM_fin_2 = LSTM(300,return_sequences=False,stateful=False,activation='softsign',kernel_initializer=kernel_init,bias_initializer=kernel_init)(LSTM_fin)


    Dense_fin = Dense(250,activation='softsign',kernel_initializer=kernel_init,bias_initializer=kernel_init)(LSTM_fin_2)

    Dense_fin_1 = Dense(200, activation='softsign', kernel_initializer=kernel_init, bias_initializer=kernel_init)(
        Dense_fin)
#




    output = tf.keras.layers.Dense(5,activation='softsign',kernel_initializer=kernel_init,bias_initializer=kernel_init)(Dense_fin_1)


    lstm_model = tf.keras.Model(inputs=input, outputs=output)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.001,
        decay_steps=55,
        decay_rate=0.975,
        staircase=True)


    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    lstm_model.compile(loss=[custom_cosine_similarity], optimizer=optimizer)
    #lstm_model.compile(loss=[custom_cosine_similarity,custom_cosine_similarity,custom_cosine_similarity,custom_cosine_similarity,custom_cosine_similarity], optimizer=optimizer)
    return lstm_model
