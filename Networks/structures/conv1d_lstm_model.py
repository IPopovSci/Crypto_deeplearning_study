import tensorflow as tf

from tensorflow.keras.layers import Input, LayerNormalization, Conv2D, \
    Flatten, MaxPooling2D
from keras.layers.convolutional_recurrent import ConvLSTM1D

from pipeline.pipelineargs import PipelineArgs
from Networks.network_config import NetworkParams
from Networks.losses_metrics import ohlcv_mse, ohlcv_cosine_similarity, metric_signs_close, ohlcv_combined

pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()


def convlstm_model():
    batch_size = pipeline_args.args['batch_size']
    time_steps = pipeline_args.args['time_steps']
    num_features = pipeline_args.args['num_features']
    extra_dim = 1

    input = Input(shape=(time_steps, num_features, 1), batch_size=batch_size)

    regularizer = None#tf.keras.regularizers.l1_l2(l1=network_args.network['l1_reg'], l2=network_args.network['l2_reg'])
    initializer = tf.keras.initializers.glorot_uniform()
    dropout = network_args.network['dropout']

    activation = tf.keras.activations.swish

    # x = Conv2D(kernel_size=[3, 3], filters=32, kernel_initializer=initializer, kernel_regularizer=regularizer,
    #            bias_initializer=initializer, bias_regularizer=regularizer, activity_regularizer=regularizer,
    #            activation=activation, padding='same')(input)

    x = ConvLSTM1D(64, stateful=False, kernel_size=3, bias_regularizer=regularizer,
                   activity_regularizer=regularizer, recurrent_regularizer=regularizer,
                   recurrent_initializer=initializer, activation=activation, kernel_initializer=initializer,
                   kernel_regularizer=regularizer, return_sequences=True, padding='same')(input)

    # x = ConvLSTM1D(64, stateful=False, kernel_size=3, bias_regularizer=regularizer,
    #                activity_regularizer=regularizer, recurrent_regularizer=regularizer,
    #                recurrent_initializer=initializer, activation=activation, kernel_initializer=initializer,
    #                kernel_regularizer=regularizer, return_sequences=True, padding='same')(x)
    #
    # x = ConvLSTM1D(64, stateful=False, kernel_size=3, bias_regularizer=regularizer,
    #                activity_regularizer=regularizer, recurrent_regularizer=regularizer,
    #                recurrent_initializer=initializer, activation=activation, kernel_initializer=initializer,
    #                kernel_regularizer=regularizer, return_sequences=True, padding='same')(x)

    x = LayerNormalization()(x)

    # x = ConvLSTM1D(64, stateful=False, kernel_size=3, bias_regularizer=regularizer,
    #                activity_regularizer=regularizer, recurrent_regularizer=regularizer,
    #                recurrent_initializer=initializer, activation=activation, kernel_initializer=initializer,
    #                kernel_regularizer=regularizer, return_sequences=True, padding='same')(x)
    #
    # x = LayerNormalization()(x)
    #
    # x = ConvLSTM1D(64, stateful=False, kernel_size=3, bias_regularizer=regularizer,
    #                activity_regularizer=regularizer, recurrent_regularizer=regularizer,
    #                recurrent_initializer=initializer, activation=activation, kernel_initializer=initializer,
    #                kernel_regularizer=regularizer, return_sequences=True, padding='same')(x)
    #
    # x = LayerNormalization()(x)


    x = Conv2D(kernel_size=[3, 3], filters=64, kernel_initializer=initializer, kernel_regularizer=regularizer,
               bias_regularizer=regularizer, activity_regularizer=regularizer,
               activation=activation, padding='same')(x)

    x = MaxPooling2D(pool_size=(2,2), activity_regularizer=regularizer)(x)


    x = Flatten()(x)

    x = LayerNormalization()(x)

    output = tf.keras.layers.Dense(5, activation='softsign', activity_regularizer=regularizer,
                                   kernel_regularizer=regularizer, bias_regularizer=regularizer,
                                   kernel_initializer=initializer)(x)

    lstm_model = tf.keras.Model(inputs=input, outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=network_args.network['lr'], amsgrad=True)

    lstm_model.compile(
        loss=profit_ratio_assymetric, optimizer=optimizer, metrics=[metric_signs_close, ohlcv_cosine_similarity, ohlcv_mse,metric_profit_ratio])

    return lstm_model
