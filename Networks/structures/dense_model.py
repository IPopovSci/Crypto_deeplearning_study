import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, GaussianNoise, AlphaDropout, Dropout, LayerNormalization, \
    BatchNormalization, LSTM

from pipeline.pipelineargs import PipelineArgs
from Networks.network_config import NetworkParams
from Networks.losses_metrics import ohlcv_mse, ohlcv_cosine_similarity, metric_signs_close, ohlcv_combined, \
    assymetric_loss, assymetric_combined, metric_loss

pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()


def dense_model():
    batch_size = pipeline_args.args['batch_size']
    time_steps = pipeline_args.args['time_steps']
    num_features = pipeline_args.args['num_features']

    input = Input(batch_shape=(batch_size, time_steps, num_features))

    regularizer = tf.keras.regularizers.l1_l2(l1=network_args.network['l1_reg'], l2=network_args.network['l2_reg'])
    initializer = tf.keras.initializers.LecunNormal()
    dropout = network_args.network['dropout']

    activation = 'selu'

    x = Dense(25, activation=activation, activity_regularizer=regularizer, kernel_regularizer=regularizer,
              bias_regularizer=regularizer, kernel_initializer=initializer, bias_initializer=initializer)(input)

    x = Dense(15, activation=activation, activity_regularizer=regularizer, kernel_regularizer=regularizer,
              bias_regularizer=regularizer, kernel_initializer=initializer, bias_initializer=initializer)(x)

    x = Dense(10, activation=activation, activity_regularizer=regularizer, kernel_regularizer=regularizer,
              bias_regularizer=regularizer, kernel_initializer=initializer, bias_initializer=initializer)(x)

    x = Dense(5, activation=activation, activity_regularizer=regularizer, kernel_regularizer=regularizer,
              bias_regularizer=regularizer, kernel_initializer=initializer, bias_initializer=initializer)(x)

    output = tf.keras.layers.Dense(5, activation='linear', activity_regularizer=regularizer,
                                   kernel_regularizer=regularizer, bias_regularizer=regularizer,
                                   kernel_initializer=initializer, bias_initializer=initializer)(x)

    lstm_model = tf.keras.Model(inputs=input, outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=network_args.network['lr'], amsgrad=True)

    lstm_model.compile(
        loss=metric_loss, optimizer=optimizer, metrics=[metric_signs_close, ohlcv_cosine_similarity, ohlcv_mse])

    return lstm_model