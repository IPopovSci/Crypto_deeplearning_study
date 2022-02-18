from tensorflow.keras.layers import LSTM,Conv1D,MaxPooling1D,Flatten, Concatenate,Dense, Input,TimeDistributed,GRU,Dropout,Bidirectional,SimpleRNN,LayerNormalization,BatchNormalization,LeakyReLU,PReLU,GaussianNoise,Convolution1D,MaxPooling1D
from Arguments import args
from LSTM.callbacks import mean_squared_error_custom,custom_cosine_similarity,metric_signs,metric_signs_loss,custom_mean_absolute_error,stock_loss,stock_loss_metric
import tensorflow as tf
from keras_self_attention import SeqSelfAttention
from tensorflow.keras import initializers
from keras_multi_head import MultiHead,MultiHeadAttention

# tf.keras.activations.swish

def create_lstm_model(x_t):
    BATCH_SIZE = args['batch_size']
    TIME_STEPS = args['time_steps']
    n_components = args['n_components']

    input = Input(batch_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]))
    regularizer = tf.keras.regularizers.l1_l2(l1=0.01,l2=0.01)
    kernel_init = tf.keras.initializers.LecunNormal()
    dropout = 0.2

# #This is First side-chain: input>LSTM(stateful)>LSTM(stateful)>TD Dense layer. The output is a 3d vector

    activation = 'selu'
    noise = GaussianNoise(0.005)(input)

    # gru = LSTM(60,return_sequences=True,stateful=True,kernel_initializer=kernel_init,activation=activation,kernel_regularizer=regularizer)(input)
    #
    # # norm_1 = LayerNormalization()(gru)
    # #
    # # leaky_1 = tf.keras.activations.selu(norm_1)
    #
    # gru_1 = LSTM(30,return_sequences=True,stateful=True,kernel_initializer=kernel_init,activation=activation,kernel_regularizer=regularizer)(gru)
    #
    # # # norm_1 = LayerNormalization()(gru)
    # # #
    # # # leaky_1 = tf.keras.activations.selu(norm_1)
    #
    # gru = LSTM(60,return_sequences=True,stateful=False,kernel_initializer=kernel_init,activation=activation,kernel_regularizer=regularizer)(input)
    #
    # att = SeqSelfAttention(units=45,kernel_regularizer=regularizer)(gru)
    #
    # #
    #
    # # conv = Conv1D(kernel_size=12,filters=128,kernel_initializer=kernel_init,kernel_regularizer=regularizer,activation=activation,data_format='channels_first')(input)
    # #
    # # conv_2 = Conv1D(kernel_size=12,filters=64,kernel_initializer=kernel_init,kernel_regularizer=regularizer,activation=activation,data_format='channels_first')(conv)
    # #
    # # conv_3 = Conv1D(kernel_size=12, filters=32, kernel_initializer=kernel_init, kernel_regularizer=regularizer,
    # #                 activation=activation,data_format='channels_first')(conv_2)
    # #
    # # pool = MaxPooling1D(pool_size=3)(conv_3)
    #
    # concat = Concatenate()([gru_1,att,pool])
    #
    # dense = TimeDistributed(Dense(65,kernel_initializer=kernel_init,activation=activation,kernel_regularizer=regularizer))(concat)
    # # #
    # gru = LSTM(61, return_sequences=False, stateful=True, kernel_initializer=kernel_init, activation=activation,kernel_regularizer=regularizer)(dense)

    conv = Conv1D(kernel_size=32,filters=128,kernel_initializer=kernel_init,kernel_regularizer=regularizer,activation=activation,padding='causal')(noise)

    #lstm = LSTM(64, return_sequences=True, stateful=True, kernel_initializer=kernel_init, activation=activation,kernel_regularizer=regularizer)(conv)

    conv_2 = Conv1D(kernel_size=16,filters=64,kernel_initializer=kernel_init,kernel_regularizer=regularizer,activation=activation,padding='causal')(conv)

    #lstm = LSTM(32, return_sequences=True, stateful=True, kernel_initializer=kernel_init, activation=activation,kernel_regularizer=regularizer)(conv_2)

    conv_3 = Conv1D(kernel_size=8, filters=64, kernel_initializer=kernel_init, kernel_regularizer=regularizer,
                    activation=activation,padding='causal')(conv_2)

    conv_4 = Conv1D(kernel_size=4, filters=32, kernel_initializer=kernel_init, kernel_regularizer=regularizer,
                    activation=activation,padding='causal')(conv_3)

    # conv_4 = Conv1D(kernel_size=4, filters=16, kernel_initializer=kernel_init, kernel_regularizer=regularizer,
    #                 activation=activation,padding='causal')(conv_3)

    #lstm = LSTM(8, return_sequences=True, stateful=True, kernel_initializer=kernel_init, activation=activation,kernel_regularizer=regularizer)(conv_3)

    pool = MaxPooling1D(pool_size=4)(conv_4)

    flat = Flatten()(pool)
    # lstm = LSTM(32, return_sequences=True, stateful=True, kernel_initializer=kernel_init, activation=activation,kernel_regularizer=regularizer)(conv_3)
    #
    # lstm = LSTM(16, return_sequences=False, stateful=False, kernel_initializer=kernel_init, activation=activation,kernel_regularizer=regularizer)(lstm)


    dense = Dense(64,kernel_initializer=kernel_init,activation=activation,kernel_regularizer=regularizer)(flat) #do we need tanh activation here? Ensemble with none mb

    # norm = LayerNormalization()(dense)
    #
    # leaky = tf.keras.activations.selu(norm)

    dense = Dense(32,kernel_initializer=kernel_init,activation=activation,kernel_regularizer=regularizer)(dense) #do we need tanh activation here? Ensemble with none mb

    # norm = LayerNormalization()(dense)
    #
    # leaky = tf.keras.activations.selu(norm)


    output = tf.keras.layers.Dense(1,activation='linear',kernel_regularizer=regularizer)(dense)


    lstm_model = tf.keras.Model(inputs=input, outputs=output)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.000001,
        decay_steps=387,
        decay_rate=0.99,
        staircase=True)

    #optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01,amsgrad=True,clipnorm=0.5,clipvalue=0.5)
    #optimizer = tf.keras.optimizers.SGD(lr=0.005,momentum=True,nesterov=True)
    #lstm_model.compile(loss=[mean_squared_error_custom], optimizer=optimizer)
    #lstm_model.compile(loss=[custom_cosine_similarity,custom_cosine_similarity,custom_cosine_similarity,custom_cosine_similarity,custom_cosine_similarity], optimizer=optimizer,metrics=metric_signs)
    lstm_model.compile(
        loss=[custom_cosine_similarity], optimizer=optimizer, metrics=metric_signs)
    #lstm_model.compile(
        #loss='CosineSimilarity', optimizer=optimizer,metrics=metric_signs)
    return lstm_model
