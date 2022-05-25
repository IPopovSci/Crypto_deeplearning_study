import tensorflow as tf

from tensorflow.keras.layers import Input, LayerNormalization, Conv2D, \
    Flatten, MaxPooling2D, Dense, GaussianNoise,Bidirectional,Concatenate
from keras.layers.convolutional_recurrent import ConvLSTM1D
from keras_self_attention import SeqSelfAttention
from pipeline.pipelineargs import PipelineArgs
from Networks.network_config import NetworkParams
from Networks.losses_metrics import ohlcv_mse, ohlcv_cosine_similarity, metric_signs_close, ohlcv_combined, \
    profit_ratio_assymetric, metric_profit_ratio
from Networks.custom_activation import p_swish,p_softsign

pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()

'''Builds a convolutional LSTM model.
Contains the structure of the model inside, along with optimizer.'''
def convlstm_model():
    batch_size = pipeline_args.args['batch_size']
    time_steps = pipeline_args.args['time_steps']
    num_features = pipeline_args.args['num_features']

    input = Input(shape=(time_steps, num_features, 1), batch_size=batch_size)

    regularizer = None  # tf.keras.regularizers.l1_l2(l1=network_args.network['l1_reg'], l2=network_args.network['l2_reg'])
    initializer = tf.keras.initializers.glorot_uniform()

    activation = tf.keras.activations.swish

    #noise = GaussianNoise(0.1)(input)

    x = ConvLSTM1D(32, stateful=True, kernel_size=3, bias_regularizer=regularizer,
                   activity_regularizer=regularizer, recurrent_regularizer=regularizer,
                   recurrent_initializer=initializer, activation=activation, kernel_initializer=initializer,
                   kernel_regularizer=regularizer, return_sequences=True, padding='same')(input)



    # #
    x = Flatten()(x)

    x = LayerNormalization()(x)

    y = ConvLSTM1D(32, stateful=False, kernel_size=3, bias_regularizer=regularizer,
                   activity_regularizer=regularizer, recurrent_regularizer=regularizer,
                   recurrent_initializer=initializer, activation=activation, kernel_initializer=initializer,
                   kernel_regularizer=regularizer, return_sequences=False, padding='same')(input)

    y = LayerNormalization()(y)

    y = SeqSelfAttention(units=32)(y)

    y = Flatten()(y)

    concat = Concatenate()([x,y])

    x = LayerNormalization()(concat)

    x = Dense(32, activation=activation, activity_regularizer=regularizer, kernel_regularizer=regularizer,
              bias_regularizer=regularizer, kernel_initializer=initializer)(
        x)

    #
    x = LayerNormalization()(x)

    output = tf.keras.layers.Dense(5, activation='linear', activity_regularizer=regularizer,
                                   kernel_regularizer=regularizer, bias_regularizer=regularizer,
                                   kernel_initializer=initializer)(x)

    lstm_model = tf.keras.Model(inputs=input, outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=network_args.network['lr'], amsgrad=True)

    lstm_model.compile(
        loss=profit_ratio_assymetric, optimizer=optimizer,
        metrics=[metric_signs_close, ohlcv_cosine_similarity, ohlcv_mse, metric_profit_ratio])

    return lstm_model
