import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, GaussianNoise, Conv1D, MaxPooling1D, Flatten, BatchNormalization, \
    Dropout

from pipeline.pipelineargs import PipelineArgs
from Networks.network_config import NetworkParams
from Networks.losses_metrics import ohlcv_mse, ohlcv_cosine_similarity, metric_signs_close, ohlcv_combined, \
    assymetric_loss, assymetric_combined, metric_loss,profit_ratio_assymetric,metric_profit_ratio

pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()


'''Builds a 1d convolutional model.
Contains the structure of the model inside, along with optimizer.'''
def conv1d_model():
    batch_size = pipeline_args.args['batch_size']
    time_steps = pipeline_args.args['time_steps']
    num_features = pipeline_args.args['num_features']

    input = Input(batch_shape=(batch_size, time_steps, num_features))

    regularizer = None  # tf.keras.regularizers.l1_l2(l1=network_args.network['l1_reg'], l2=network_args.network['l2_reg'])
    initializer = tf.keras.initializers.glorot_uniform()

    activation = tf.keras.activations.swish

    # noise = GaussianNoise(0.05)(input) # Use this layer if you wish to apply gaussian noise to input
    x = Conv1D(kernel_size=5, filters=128, kernel_initializer=initializer, kernel_regularizer=regularizer,
               bias_regularizer=regularizer, activity_regularizer=regularizer,
               activation=activation, padding='same')(input)

    x = BatchNormalization()(x)

    x = Conv1D(kernel_size=3, filters=128, kernel_initializer=initializer, kernel_regularizer=regularizer,
               bias_regularizer=regularizer, activity_regularizer=regularizer,
               activation=activation, padding='same')(x)

    x = BatchNormalization()(x)

    x = Conv1D(kernel_size=3, filters=128, kernel_initializer=initializer, kernel_regularizer=regularizer,
               bias_regularizer=regularizer, activity_regularizer=regularizer,
               activation=activation, padding='same')(x)

    x = BatchNormalization()(x)

    x = Dense(128, activation=activation, activity_regularizer=regularizer, kernel_regularizer=regularizer,
              bias_regularizer=regularizer, kernel_initializer=initializer)(
        x)

    x = BatchNormalization()(x)

    output = tf.keras.layers.Dense(5, activation='softsign', activity_regularizer=regularizer,
                                   kernel_regularizer=regularizer, bias_regularizer=regularizer,
                                   kernel_initializer=initializer)(x)

    lstm_model = tf.keras.Model(inputs=input, outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=network_args.network['lr'], amsgrad=True)

    lstm_model.compile(
        loss=profit_ratio_assymetric, optimizer=optimizer,
        metrics=[metric_signs_close, ohlcv_cosine_similarity, ohlcv_mse, metric_profit_ratio])

    return lstm_model
