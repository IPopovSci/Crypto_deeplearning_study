import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, GaussianNoise, LSTM, TimeDistributed, LayerNormalization,BatchNormalization,Dropout,AlphaDropout

from keras_self_attention import SeqSelfAttention
from pipeline.pipelineargs import PipelineArgs
from Networks.network_config import NetworkParams
from Networks.losses_metrics import ohlcv_mse, ohlcv_cosine_similarity, metric_signs_close, ohlcv_combined

pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()

'''This one is based on the article about time series and self attention'''


def lstm_att_model():
    batch_size = pipeline_args.args['batch_size']
    time_steps = pipeline_args.args['time_steps']
    num_features = pipeline_args.args['num_features']

    input = Input(batch_shape=(batch_size, time_steps, num_features))

    regularizer = tf.keras.regularizers.l1_l2(l1=network_args.network['l1_reg'], l2=network_args.network['l2_reg'])
    initializer = tf.keras.initializers.LecunNormal()
    dropout = network_args.network['dropout']
    bias_initializer = initializer

    activation = tf.keras.activations.swish

    # #This is First side-chain: input>LSTM(stateful)>LSTM(stateful)>TD Dense layer. The output is a 3d vector
    x = LSTM(int(75),return_sequences=True, stateful=False, activation=activation,kernel_initializer=initializer,bias_initializer=initializer,activity_regularizer=regularizer,kernel_regularizer=regularizer,bias_regularizer=regularizer)(input)

    x = LayerNormalization()(x)

    #x = Dropout(dropout)(x)

    x = LSTM(int(50), return_sequences=True, stateful=False, activation=activation,kernel_initializer=initializer,bias_initializer=initializer,activity_regularizer=regularizer,kernel_regularizer=regularizer,bias_regularizer=regularizer)(x)

    x = LayerNormalization()(x)

    #x = AlphaDropout(dropout)(x)

    x = Dense(40, activation=activation,kernel_initializer=initializer,bias_initializer=initializer,activity_regularizer=regularizer,kernel_regularizer=regularizer,bias_regularizer=regularizer)(
        x)


    x = LayerNormalization()(x)

    #x = Dropout(dropout)(x)

    x = Dense(25, activation=activation,activity_regularizer=regularizer,kernel_regularizer=regularizer,bias_regularizer=regularizer,kernel_initializer=initializer,bias_initializer=initializer)(
        x)

    x = LayerNormalization()(x)

    #x = Dropout(dropout)(x)


    output = tf.keras.layers.Dense(5, activation=activation,kernel_initializer=initializer,bias_initializer=initializer)(x)

    lstm_model = tf.keras.Model(inputs=input, outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=network_args.network['lr'], amsgrad=True)

    lstm_model.compile(
        loss=ohlcv_combined, optimizer=optimizer, metrics=[metric_signs_close, ohlcv_cosine_similarity])

    return lstm_model
