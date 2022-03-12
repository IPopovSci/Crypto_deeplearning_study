import tensorflow as tf
from tensorflow.keras.layers import AlphaDropout,Conv2D,LSTM,Conv1D,MaxPooling1D,Flatten, Concatenate,Dense, Input,TimeDistributed,GRU,Dropout,Bidirectional,SimpleRNN,LayerNormalization,BatchNormalization,LeakyReLU,PReLU,GaussianNoise,Convolution1D,MaxPooling1D
from Arguments import args
from keras.layers.convolutional_recurrent import ConvLSTM1D
from training.callbacks import custom_cosine_similarity,metric_signs,custom_mean_absolute_error,portfolio_metric

from keras.layers.convolutional_recurrent import ConvLSTM1D
from keras_self_attention import SeqSelfAttention
from keras import initializers
from keras_multi_head import MultiHead,MultiHeadAttention

# tf.keras.activations.swish

def create_convlstm_model(x_t):
    BATCH_SIZE = args['batch_size']
    TIME_STEPS = args['time_steps']
    n_components = args['n_components']

    input = Input(shape=(TIME_STEPS, x_t.shape[2],x_t.shape[3]),batch_size=BATCH_SIZE)
    money = Input(shape=(1),batch_size=BATCH_SIZE, name='money')
    y_true = Input(shape=(5),batch_size=BATCH_SIZE)
    regularizer = None#tf.keras.regularizers.l2(l2=0.0005)
    kernel_init = tf.keras.initializers.LecunNormal()
    dropout = 0.2
    '''Net of conv1dlstm w/ conv2d at the end produces ALL the steps i.e if we set time-step to 10, and prediction interval to 10, we get all 10 points in the future! Wow this might be new'''
# #This is First side-chain: input>training(stateful)>training(stateful)>TD Dense layer. The output is a 3d vector

    activation = 'softsign'
    noise = GaussianNoise(0.001)(input)

    convlstm = ConvLSTM1D(64,stateful=True,kernel_size=3,recurrent_initializer=kernel_init,activation=activation,kernel_initializer=kernel_init,kernel_regularizer=regularizer,return_sequences=True,padding='same')(noise)

    convlstm = BatchNormalization()(convlstm)
    #
    #convlstm = AlphaDropout(0.15)(convlstm)
    #
    # #convlstm = ConvLSTM1D(64,stateful=True,kernel_size=3,recurrent_initializer=kernel_init,activation=activation,kernel_initializer=kernel_init, kernel_regularizer=regularizer,return_sequences=True,padding='same')(convlstm)
    #
    convlstm = Conv2D(64,kernel_size=[1,1])(convlstm)

    convlstm = BatchNormalization()(convlstm)
    #
    #convlstm = AlphaDropout(0.15)(convlstm)
    #
    convlstm = ConvLSTM1D(64,stateful=True,kernel_size=3,recurrent_initializer=kernel_init, activation=activation,kernel_initializer=kernel_init, kernel_regularizer=regularizer,return_sequences=False,padding='same')(convlstm)
    # #
    # convlstm = LayerNormalization()(convlstm)
    #
    # convlstm = AlphaDropout(0.3)(convlstm)

    # #convlstm  = Conv1D(64,kernel_size=3, activation=activation,kernel_initializer=kernel_init, kernel_regularizer=regularizer,padding='same')(convlstm)
    # # #
    # convlstm = LayerNormalization()(convlstm)
    #
    # convlstm = AlphaDropout(0.5)(convlstm)
    #This shouldn't be stateful, should only get last day results multiple time, not time steps of previous ones
    output = GRU(32,reset_after=False,stateful=True,return_sequences=False,recurrent_initializer=kernel_init, activation=activation,kernel_initializer=kernel_init, kernel_regularizer=regularizer)(convlstm)

    output = BatchNormalization()(output)

    #output = AlphaDropout(0.15)(output)

    '''output above is (128, 25, 61, 1)'''
    #output = LSTM(1, activation=activation, kernel_regularizer=regularizer,return_sequences=False)(convlstm)
    #
    # convlstm = Conv2D(1, kernel_size=(1, 1), activation=activation, kernel_regularizer=regularizer)(convlstm)

    #output = Flatten()(output)


    output = Dense(8,activation=activation)(output)


    y_pred = tf.keras.layers.Dense(5,activation='softsign',kernel_regularizer=regularizer)(output)


    lstm_model = tf.keras.Model(inputs=[input,y_true], outputs=y_pred)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.000001,
        decay_steps=387,
        decay_rate=0.99,
        staircase=True)

    #optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    #optimizer = tf.keras.optimizers.SGD(lr=0.005,momentum=True,nesterov=True)
    #lstm_model.compile(loss=[mean_squared_error_custom], optimizer=optimizer)
    # TODO portfolio_metric
    # lstm_model.add_metric(portfolio_metric(y_true,y_pred,money),name='portfolio_metric')
    #lstm_model.compile(loss=[custom_cosine_similarity,custom_cosine_similarity,custom_cosine_similarity,custom_cosine_similarity,custom_cosine_similarity], optimizer=optimizer,metrics=metric_signs)
    #lstm_model.add_metric(portfolio_metric,name='portfolio_metric',aggregation='mean')
    lstm_model.compile(
        loss=[custom_cosine_similarity,'mse'], optimizer=optimizer, metrics=[metric_signs])

    #lstm_model.compile(
        #loss='CosineSimilarity', optimizer=optimizer,metrics=metric_signs)
    return lstm_model
