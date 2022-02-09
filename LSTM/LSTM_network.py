from tensorflow.keras.layers import LSTM, Dense, Input,TimeDistributed,GRU,Dropout,Bidirectional,SimpleRNN,LayerNormalization,BatchNormalization,LeakyReLU,PReLU,GaussianNoise,Convolution1D,MaxPooling1D
from Arguments import args
from LSTM.callbacks import mean_squared_error_custom,custom_cosine_similarity,metric_signs,custom_mean_absolute_error,stock_loss
import tensorflow as tf
from keras_self_attention import SeqSelfAttention
from tensorflow.keras import initializers
from keras_multi_head import MultiHead,MultiHeadAttention



def create_lstm_model(x_t):
    BATCH_SIZE = args['batch_size']
    TIME_STEPS = args['time_steps']
    n_components = args['n_components']

    input = Input(batch_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]))
    regularizer = tf.keras.regularizers.l2(0.01)
    kernel_init = initializers.glorot_uniform()
    dropout = 0.5

# #This is First side-chain: input>LSTM(stateful)>LSTM(stateful)>TD Dense layer. The output is a 3d vector

    activation = 'tanh'
    gru = Bidirectional(GRU(30,return_sequences=False,stateful=True,reset_after=False))(input)

    # gru =GRU(45,return_sequences=False,stateful=False,reset_after=False)(gru)
    #
    # dense = Dense(30,activation=activation)(gru)

    output = tf.keras.layers.Dense(3,activation='softmax',kernel_initializer=kernel_init,bias_initializer=kernel_init)(gru)



    lstm_model = tf.keras.Model(inputs=input, outputs=output)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.0001,
        decay_steps=10000,
        decay_rate=0.98,
        staircase=True)

    #optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,amsgrad=True)
    #optimizer = tf.keras.optimizers.SGD(lr=0.00001,momentum=True,nesterov=True)
    #lstm_model.compile(loss=[mean_squared_error_custom], optimizer=optimizer)
    #lstm_model.compile(loss=[custom_cosine_similarity,custom_cosine_similarity,custom_cosine_similarity,custom_cosine_similarity,custom_cosine_similarity], optimizer=optimizer,metrics=metric_signs)
    lstm_model.compile(
        loss='categorical_crossentropy', optimizer=optimizer, metrics='accuracy')
    #lstm_model.compile(
        #loss='CosineSimilarity', optimizer=optimizer,metrics=metric_signs)
    return lstm_model
