import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, GaussianNoise, Conv1D, MaxPooling1D, Flatten, BatchNormalization, \
    Dropout, Conv2D, MaxPooling2D, LSTM, AlphaDropout, DepthwiseConv2D, SeparableConv2D, LocallyConnected2D, \
    TimeDistributed

from pipeline.pipelineargs import PipelineArgs
from Networks.network_config import NetworkParams
from Networks.losses_metrics import ohlcv_mse, ohlcv_cosine_similarity, metric_signs_close, ohlcv_combined, \
    assymetric_loss, assymetric_combined, metric_loss, profit_ratio_cosine, profit_ratio_assymetric, metric_profit_ratio
from Networks.custom_activation import cyclemoid
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from keras_self_attention import SeqSelfAttention
from Networks.custom_activation import p_swish,p_softsign

pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()

get_custom_objects().update({'cyclemoid': Activation(cyclemoid)})

'''Builds a 2d convolutional model.
Contains the structure of the model inside, along with optimizer.'''
def conv2d_model():
    batch_size = pipeline_args.args['batch_size']
    print(batch_size)
    time_steps = pipeline_args.args['time_steps']
    num_features = pipeline_args.args['num_features']

    input = Input(shape=(time_steps, num_features, 1), batch_size=batch_size)

    regularizer = None#tf.keras.regularizers.l1_l2(l1=network_args.network['l1_reg'], l2=network_args.network['l2_reg'])
    initializer = tf.keras.initializers.glorot_uniform()

    activation = 'linear'  # tf.keras.activations.swish

    #x = GaussianNoise(0.001)(input) #Use this as first layer for gaussian noise

    x = Conv2D(kernel_size=[3, 3], filters=64, kernel_initializer=initializer, kernel_regularizer=regularizer,
               bias_regularizer=regularizer, activity_regularizer=regularizer,
               activation=activation, padding='same')(input)



    x = p_swish()(x)

    x = BatchNormalization()(x)



    x = MaxPooling2D(pool_size=(2, 2), activity_regularizer=regularizer)(x)

    x = Conv2D(kernel_size=[3, 3], filters=64, kernel_initializer=initializer, kernel_regularizer=regularizer
               , bias_regularizer=regularizer, activity_regularizer=regularizer,
               activation=activation, padding='same')(x)



    x = p_swish()(x)

    x = BatchNormalization()(x)



    x = MaxPooling2D(pool_size=(2, 2), activity_regularizer=regularizer)(x)

    x = Conv2D(kernel_size=[3, 3], filters=64, kernel_initializer=initializer, kernel_regularizer=regularizer,
               bias_regularizer=regularizer, activity_regularizer=regularizer,
               activation=activation, padding='same')(x)



    x = p_swish()(x)

    #x = BatchNormalization()(x)



    x = MaxPooling2D(pool_size=(2, 2), activity_regularizer=regularizer)(x)

    x = Flatten()(x)

    x = BatchNormalization()(x)

    x = Dense(128, activation=activation, activity_regularizer=regularizer, kernel_regularizer=regularizer,
              bias_regularizer=regularizer, kernel_initializer=initializer)(
        x)

    x = p_swish()(x)

    x = BatchNormalization()(x)



    output = tf.keras.layers.Dense(5, activation=activation, activity_regularizer=regularizer,
                                   kernel_regularizer=regularizer, bias_regularizer=regularizer,
                                   kernel_initializer=initializer)(x)

    output = p_softsign()(output)


    lstm_model = tf.keras.Model(inputs=input, outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=network_args.network['lr'], amsgrad=True)

    lstm_model.compile(
        loss=profit_ratio_assymetric, optimizer=optimizer,
        metrics=[metric_signs_close, ohlcv_cosine_similarity, ohlcv_mse, metric_profit_ratio])

    return lstm_model
