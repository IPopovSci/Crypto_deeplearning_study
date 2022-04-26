import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, LSTM, LayerNormalization,GaussianNoise,Bidirectional,Dropout

from pipeline.pipelineargs import PipelineArgs
from Networks.network_config import NetworkParams
from Networks.losses_metrics import ohlcv_cosine_similarity, metric_signs_close, metric_loss, profit_ratio_assymetric, \
    ohlcv_mse, metric_profit_ratio,profit_ratio_cosine,assymetric_loss_mse
from Networks.custom_activation import p_swish,p_softsign

pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()

'''Builds an lstm model.
Contains the structure of the model inside, along with optimizer.'''
def lstm_att_model():
    batch_size = pipeline_args.args['batch_size']
    time_steps = pipeline_args.args['time_steps']
    num_features = pipeline_args.args['num_features']

    input = Input(batch_shape=(batch_size, time_steps, num_features))

    regularizer = None  # tf.keras.regularizers.l1_l2(l1=network_args.network['l1_reg'], l2=network_args.network['l2_reg'])
    initializer = tf.keras.initializers.glorot_uniform()

    activation = tf.keras.activations.swish

    # x = GaussianNoise(0.5)(input)

    x = LSTM(int(64), return_sequences=True, stateful=False, activation=activation, kernel_initializer=initializer)(
        input)

    #x = p_swish()(x)

    x = LayerNormalization()(x)


    # #
    # x = LSTM(int(64),dropout=0.2,recurrent_dropout=0.2, return_sequences=True, stateful=False, activation=activation, kernel_initializer=initializer)(x)
    # #
    # # #x = p_swish()(x)
    # #
    # x = LayerNormalization()(x)
    # #
    # #
    # x = LSTM(int(32),dropout=0.2,recurrent_dropout=0.2, return_sequences=True, stateful=False, activation=activation, kernel_initializer=initializer)(x)
    #
    # #x = p_swish()(x)
    #
    # x = LayerNormalization()(x)
    #
    #
    #
    # x = LSTM(int(16), return_sequences=True, stateful=False, activation=activation, kernel_initializer=initializer)(x)
    #
    # x = p_swish()(x)
    #
    # x = LayerNormalization()(x)
    #
    #
    #
    x = Dense(512, activation=activation, kernel_initializer=initializer,
              activity_regularizer=regularizer)(x)
    # # #
    # # # #x = p_swish()(x)
    # # #
    x = LayerNormalization()(x)



    output = tf.keras.layers.Dense(5, activation='linear',
                                   kernel_initializer=initializer)(x)

    #output = p_softsign()(output)

    lstm_model = tf.keras.Model(inputs=input, outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=network_args.network['lr'])

    lstm_model.compile(
        loss=profit_ratio_assymetric, optimizer=optimizer,
        metrics=[metric_signs_close, ohlcv_cosine_similarity, ohlcv_mse, metric_profit_ratio])

    return lstm_model
