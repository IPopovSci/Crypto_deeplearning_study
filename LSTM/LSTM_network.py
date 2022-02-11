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
    regularizer = None#tf.keras.regularizers.l2(l2=0.000001)
    kernel_init = initializers.glorot_uniform()
    dropout = 0.2

# #This is First side-chain: input>LSTM(stateful)>LSTM(stateful)>TD Dense layer. The output is a 3d vector

    activation = None
    noise = GaussianNoise(0.01)(input)

    gru = LSTM(50,return_sequences=True,stateful=True,dropout=dropout,recurrent_dropout=dropout,activation = activation,kernel_regularizer=regularizer,activity_regularizer=regularizer,bias_regularizer=regularizer)(noise)

    norm_1 = LayerNormalization()(gru)

    leaky_1 = tf.keras.activations.swish(norm_1)

    tddense = TimeDistributed(Dense(32))(leaky_1)

    gru = GRU(50, return_sequences=True, stateful=False, reset_after=False, dropout=dropout, recurrent_dropout=dropout,activation = activation,kernel_regularizer=regularizer,activity_regularizer=regularizer,bias_regularizer=regularizer)(noise)

    norm_2 = LayerNormalization()(gru)

    leaky_2 = tf.keras.activations.swish(norm_2)

    attention = SeqSelfAttention()(leaky_2)



    norm_2 = LayerNormalization()(attention)

    leaky_2 = tf.keras.activations.swish(norm_2)

    tddense_2 = TimeDistributed(Dense(32))(leaky_2)

    concat = Concatenate()([tddense,tddense_2])

    norm = LayerNormalization()(concat)

    leaky = tf.keras.activations.swish(norm)

    gru = LSTM(32, return_sequences=False, stateful=True, dropout=dropout, recurrent_dropout=dropout,activation = activation,kernel_regularizer=regularizer,activity_regularizer=regularizer,bias_regularizer=regularizer)(leaky)

    norm = LayerNormalization()(gru)

    leaky = tf.keras.activations.swish(norm)

    dense = Dense(12,activation=activation,kernel_regularizer=regularizer,activity_regularizer=regularizer,bias_regularizer=regularizer)(leaky) #do we need tanh activation here? Ensemble with none mb

    norm = LayerNormalization()(dense)

    leaky = tf.keras.activations.swish(norm)


    output = tf.keras.layers.Dense(1,activation='linear',kernel_initializer=kernel_init,bias_initializer=kernel_init,kernel_regularizer=regularizer,activity_regularizer=regularizer,bias_regularizer=regularizer)(leaky)


    lstm_model = tf.keras.Model(inputs=input, outputs=output)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.00001,
        decay_steps=387,
        decay_rate=0.99,
        staircase=True)

    #optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule,amsgrad=True,clipnorm=0.99)
    #optimizer = tf.keras.optimizers.SGD(lr=0.0000005,momentum=True,nesterov=True)
    #lstm_model.compile(loss=[mean_squared_error_custom], optimizer=optimizer)
    #lstm_model.compile(loss=[custom_cosine_similarity,custom_cosine_similarity,custom_cosine_similarity,custom_cosine_similarity,custom_cosine_similarity], optimizer=optimizer,metrics=metric_signs)
    lstm_model.compile(
        loss='cosine_similarity', optimizer=optimizer, metrics=metric_signs)
    #lstm_model.compile(
        #loss='CosineSimilarity', optimizer=optimizer,metrics=metric_signs)
    return lstm_model
