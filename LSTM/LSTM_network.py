from tensorflow.keras.layers import LSTM, Dense, Input,TimeDistributed,GRU,Dropout,Bidirectional,LayerNormalization,BatchNormalization,LeakyReLU,PReLU
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
    regularizer = None #tf.keras.regularizers.l1_l2(1e-3)
    kernel_init = initializers.glorot_uniform()

# #This is First side-chain: input>LSTM(stateful)>LSTM(stateful)>TD Dense layer. The output is a 3d vector
    norm_inp = input
    activation = 'tanh'



    LSTM_1 = LSTM(int(100),kernel_regularizer=regularizer,activity_regularizer=regularizer,bias_regularizer=regularizer, return_sequences=True, stateful=True,activation=activation,kernel_initializer=kernel_init,bias_initializer=kernel_init,dropout=0.3,recurrent_dropout=0.3)(norm_inp)

    norm = BatchNormalization()(LSTM_1)

    # leak = PReLU()(norm)


# # #
    #LSTM_2 = Bidirectional(LSTM(int(61), return_sequences=True, stateful=False,activation='softsign',kernel_initializer=kernel_init,bias_initializer=kernel_init,dropout=0.6,recurrent_dropout=0.6))(norm)

    #norm_2 = LayerNormalization()(LSTM_1)

    # Dense_1 = TimeDistributed(Dense(45,activation=activation,kernel_initializer=kernel_init,bias_initializer=kernel_init))(LSTM_1)
# #This is the attention side-chain: LSTM(Stateless)>LSTM>Attention. The output is a 3d vector
    LSTM_3 = LSTM(int(80), return_sequences=False, stateful=False,activation=activation,kernel_initializer=kernel_init,bias_initializer=kernel_init,dropout=0.3,recurrent_dropout=0.3)(norm)

    norm = BatchNormalization()(LSTM_3)

    # leak = PReLU()(norm)
#
#
# #     #LSTM_4 = LSTM(int(75), return_sequences=True, stateful=False,activation='softsign',kernel_initializer=kernel_init,bias_initializer=kernel_init,dropout=0.2,recurrent_dropout=0.2)(LSTM_3)
# #
#
# # # This is the attention side-chain: LSTM(Stateless)>LSTM>Attention. The output is a 3d vector
# #     LSTM_5 = LSTM(int(50), return_sequences=True, stateful=False, activation=activation,kernel_initializer=kernel_init,bias_initializer=kernel_init,dropout=0,recurrent_dropout=0)(norm_inp)
#
#     norm = LayerNormalization()(LSTM_3)
#
    # attention_1 = SeqSelfAttention(units=45, attention_type='additive', attention_activation=activation,
    #                                kernel_initializer=kernel_init, bias_initializer=kernel_init)(LSTM_3)
#
#     # LSTM_6 = LSTM(int(75), return_sequences=True, stateful=False, activation='softsign',kernel_initializer=kernel_init,bias_initializer=kernel_init)(LSTM_5)
#     #norm = LayerNormalization()(LSTM_5)
#     # attention_2 = SeqSelfAttention(units=45,attention_activation=activation,attention_type='multiplicative',kernel_initializer=kernel_init,bias_initializer=kernel_init)(LSTM_5)
# # #Concat the sidechains and provide output (5 values, 2d vector)
# # #
#     concat = tf.keras.layers.concatenate([Dense_1,attention_1])
#
#     norm = LayerNormalization()(concat)
# #
    dropout = Dropout(rate=0.3)(norm)
# #
#     LSTM_fin = LSTM(80,return_sequences=False,stateful=True,activation=activation,kernel_initializer=kernel_init,bias_initializer=kernel_init)(norm) #softsign activation for this layer works?
# #
# #     LSTM_fin_2 = Bidirectional(LSTM(170,return_sequences=True,stateful=True,activation='softsign',kernel_initializer=kernel_init,bias_initializer=kernel_init,dropout=0.7,recurrent_dropout=0.7))(dropout)
# #
#
#
    # LSTM_fin_3 = LSTM(60,kernel_regularizer=regularizer,activity_regularizer=regularizer,bias_regularizer=regularizer, return_sequences=False, stateful=False, activation=activation, kernel_initializer=kernel_init,
    #                 bias_initializer=kernel_init,dropout=0,recurrent_dropout=0)(concat)

#     # LSTM_fin = LSTM(200, return_sequences=False, stateful=False, activation='softsign', kernel_initializer=kernel_init,
#     #                 bias_initializer=kernel_init,dropout=0.3,recurrent_dropout=0.3)(LSTM_fin_3)
#
#     norm = LayerNormalization()(LSTM_fin)
#
# #
    Dense_fin = Dense(60,activation=activation,kernel_initializer=kernel_init,bias_initializer=kernel_init)(dropout)

    norm = BatchNormalization()(Dense_fin)

    # leak = PReLU()(norm)
#
# # #
#
#     norm = LayerNormalization()(Dense_fin)
#
    dropout = Dropout(rate=0.2)(norm)
#

    output = tf.keras.layers.Dense(1,activation='linear',kernel_initializer=kernel_init,bias_initializer=kernel_init)(dropout)

    #output_activation = tf.keras.layers.Activation('sigmoid')(output)


    lstm_model = tf.keras.Model(inputs=input, outputs=output)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.0001,
        decay_steps=10000,
        decay_rate=0.98,
        staircase=True)

    #optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    #optimizer = tf.keras.optimizers.SGD(lr=0.00001,momentum=True,nesterov=True)
    #lstm_model.compile(loss=[mean_squared_error_custom], optimizer=optimizer)
    #lstm_model.compile(loss=[custom_cosine_similarity,custom_cosine_similarity,custom_cosine_similarity,custom_cosine_similarity,custom_cosine_similarity], optimizer=optimizer,metrics=metric_signs)
    lstm_model.compile(
        loss=stock_loss, optimizer=optimizer, metrics=metric_signs)
    #lstm_model.compile(
        #loss='CosineSimilarity', optimizer=optimizer,metrics=metric_signs)
    return lstm_model
