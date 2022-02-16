from tensorflow.keras.layers import LSTM, Concatenate,Dense, Input,TimeDistributed,GRU,Dropout,Bidirectional,SimpleRNN,LayerNormalization,BatchNormalization,LeakyReLU,PReLU,GaussianNoise,Convolution1D,MaxPooling1D
from Arguments import args
from LSTM.callbacks import mean_squared_error_custom,custom_cosine_similarity,metric_signs,custom_mean_absolute_error,stock_loss,stock_loss_metric
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
    regularizer = tf.keras.regularizers.l1_l2(l1=0.001,l2=0.001)
    kernel_init = tf.keras.initializers.LecunNormal()
    dropout = 0.2

# #This is First side-chain: input>LSTM(stateful)>LSTM(stateful)>TD Dense layer. The output is a 3d vector

    activation = 'selu'
    noise = GaussianNoise(0.001)(input)

    gru = LSTM(125,return_sequences=True,stateful=True,kernel_initializer=kernel_init,activation=activation,kernel_regularizer=regularizer)(input)

    # norm_1 = LayerNormalization()(gru)
    #
    # leaky_1 = tf.keras.activations.selu(norm_1)

    gru_1 = LSTM(100,return_sequences=True,stateful=False,kernel_initializer=kernel_init,activation=activation,kernel_regularizer=regularizer)(gru)

    # # norm_1 = LayerNormalization()(gru)
    # #
    # # leaky_1 = tf.keras.activations.selu(norm_1)

    gru = LSTM(125,return_sequences=True,stateful=False,kernel_initializer=kernel_init,activation=activation,kernel_regularizer=regularizer)(input)

    att = SeqSelfAttention(units=100,kernel_regularizer=regularizer)(gru)
    #
    concat = Concatenate()([gru_1,att])

    dense = TimeDistributed(Dense(150,kernel_initializer=kernel_init,activation=activation,kernel_regularizer=regularizer))(concat)
    # #
    gru = LSTM(100, return_sequences=False, stateful=False, kernel_initializer=kernel_init, activation=activation,kernel_regularizer=regularizer)(dense)


    dense = Dense(45,kernel_initializer=kernel_init,activation=activation,kernel_regularizer=regularizer)(gru) #do we need tanh activation here? Ensemble with none mb

    # norm = LayerNormalization()(dense)
    #
    # leaky = tf.keras.activations.selu(norm)

    dense = Dense(20,kernel_initializer=kernel_init,activation=activation,kernel_regularizer=regularizer)(dense) #do we need tanh activation here? Ensemble with none mb

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
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001,amsgrad=True,clipvalue=0.5,clipnorm=0.5)
    #optimizer = tf.keras.optimizers.SGD(lr=0.005,momentum=True,nesterov=True)
    #lstm_model.compile(loss=[mean_squared_error_custom], optimizer=optimizer)
    #lstm_model.compile(loss=[custom_cosine_similarity,custom_cosine_similarity,custom_cosine_similarity,custom_cosine_similarity,custom_cosine_similarity], optimizer=optimizer,metrics=metric_signs)
    lstm_model.compile(
        loss=custom_cosine_similarity, optimizer=optimizer, metrics=metric_signs)
    #lstm_model.compile(
        #loss='CosineSimilarity', optimizer=optimizer,metrics=metric_signs)
    return lstm_model
