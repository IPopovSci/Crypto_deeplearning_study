
import tensorflow as tf

from tensorflow.keras.layers import LSTM,Conv1D,MaxPooling1D,Flatten, Concatenate,Dense, Input,TimeDistributed,GRU,Dropout,Bidirectional,SimpleRNN,LayerNormalization,BatchNormalization,LeakyReLU,PReLU,GaussianNoise,Convolution1D,MaxPooling1D
from Arguments import args
from training.callbacks import custom_cosine_similarity,metric_signs,custom_mean_absolute_error

from keras_self_attention import SeqSelfAttention
from tensorflow.keras import initializers
from keras_multi_head import MultiHead,MultiHeadAttention

# tf.keras.activations.swish

def create_lstm_model(x_t):
    BATCH_SIZE = args['batch_size']
    TIME_STEPS = args['time_steps']
    n_components = args['n_components']

    input = Input(batch_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]))
    regularizer = tf.keras.regularizers.l1_l2(l1=0.1,l2=0.1)
    kernel_init = tf.keras.initializers.LecunNormal()
    dropout = 0.2

# #This is First side-chain: input>training(stateful)>training(stateful)>TD Dense layer. The output is a 3d vector

    activation = 'selu'
    noise = GaussianNoise(0.05)(input)



    conv = Conv1D(kernel_size=16,filters=128,kernel_initializer=kernel_init,kernel_regularizer=regularizer,activation=activation,padding='causal')(noise)



    conv_2 = Conv1D(kernel_size=8,filters=64,kernel_initializer=kernel_init,kernel_regularizer=regularizer,activation=activation,padding='causal')(conv)

    #lstm = training(32, return_sequences=True, stateful=True, kernel_initializer=kernel_init, activation=activation,kernel_regularizer=regularizer)(conv_2)

    conv_3 = Conv1D(kernel_size=8, filters=32, kernel_initializer=kernel_init, kernel_regularizer=regularizer,
                    activation=activation,padding='causal')(conv_2)


    conv_4 = Conv1D(kernel_size=4, filters=16, kernel_initializer=kernel_init, kernel_regularizer=regularizer,
                    activation=activation,padding='causal')(conv_3)

    #lstm = training(8, return_sequences=True, stateful=True, kernel_initializer=kernel_init, activation=activation,kernel_regularizer=regularizer)(conv_3)

    pool = MaxPooling1D(pool_size=3)(conv_4)

    # lstm = training(8, return_sequences=False, stateful=True, kernel_initializer=kernel_init, activation=activation,
    #              kernel_regularizer=regularizer)(pool)


    gru = GRU(32,return_sequences=False,reset_after=False,stateful=False)(pool)
    #
    # att = SeqSelfAttention(units=32)(gru)
    #
    # conv = Concatenate()([att, pool])

    flat = Flatten()(pool)

    # lstm = training(32, return_sequences=True, stateful=True, kernel_initializer=kernel_init, activation=activation,kernel_regularizer=regularizer)(pool)
    # #
    # lstm = training(16, return_sequences=False, stateful=False, kernel_initializer=kernel_init, activation=activation,kernel_regularizer=regularizer)(lstm)



    dense = Dense(8,kernel_initializer=kernel_init,activation=activation,kernel_regularizer=regularizer)(gru) #do we need tanh activation here? Ensemble with none mb

    # norm = LayerNormalization()(dense)
    #
    # leaky = tf.keras.activations.selu(norm)

    dense = Dense(4,kernel_initializer=kernel_init,activation=activation,kernel_regularizer=regularizer)(dense) #do we need tanh activation here? Ensemble with none mb

    # norm = LayerNormalization()(dense)
    #
    # leaky = tf.keras.activations.selu(norm)
    dense = Dense(2, kernel_initializer=kernel_init, activation=activation, kernel_regularizer=regularizer)(dense)


    output = tf.keras.layers.Dense(1,activation='linear',kernel_regularizer=regularizer)(dense)


    lstm_model = tf.keras.Model(inputs=input, outputs=output)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.000001,
        decay_steps=387,
        decay_rate=0.99,
        staircase=True)

    #optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
    #optimizer = tf.keras.optimizers.SGD(lr=0.005,momentum=True,nesterov=True)
    #lstm_model.compile(loss=[mean_squared_error_custom], optimizer=optimizer)
    #lstm_model.compile(loss=[custom_cosine_similarity,custom_cosine_similarity,custom_cosine_similarity,custom_cosine_similarity,custom_cosine_similarity], optimizer=optimizer,metrics=metric_signs)
    lstm_model.compile(
        loss=[custom_cosine_similarity], optimizer=optimizer, metrics=metric_signs)
    #lstm_model.compile(
        #loss='CosineSimilarity', optimizer=optimizer,metrics=metric_signs)
    return lstm_model