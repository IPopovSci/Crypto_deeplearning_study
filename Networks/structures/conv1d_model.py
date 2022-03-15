
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, GaussianNoise,Conv1D,MaxPooling1D,Flatten

from pipeline.pipelineargs import PipelineArgs
from Networks.network_config import NetworkParams
from Networks.losses_metrics import ohlcv_mse,ohlcv_cosine_similarity,metric_signs_close,ohlcv_combined

pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()


def conv1d_model():
    batch_size = pipeline_args.args['batch_size']
    time_steps = pipeline_args.args['time_steps']
    num_features = pipeline_args.args['num_features']


    input = Input(batch_shape=(batch_size, time_steps, num_features))

    regularizer = tf.keras.regularizers.l1_l2(l1=network_args.network['l1_reg'],l2=network_args.network['l2_reg'])
    initializer = tf.keras.initializers.LecunNormal()
    dropout = network_args.network['dropout']

    activation = 'selu'

    noise = GaussianNoise(0.05)(input)

    conv = Conv1D(kernel_size=32, filters=128, kernel_initializer=initializer, kernel_regularizer=regularizer,
                  activation=activation, padding='causal')(noise)


    conv_2 = Conv1D(kernel_size=16, filters=64, kernel_initializer=initializer, kernel_regularizer=regularizer,
                    activation=activation, padding='causal')(conv)


    conv_3 = Conv1D(kernel_size=8, filters=64, kernel_initializer=initializer, kernel_regularizer=regularizer,
                    activation=activation, padding='causal')(conv_2)

    conv_4 = Conv1D(kernel_size=4, filters=32, kernel_initializer=initializer, kernel_regularizer=regularizer,
                    activation=activation, padding='causal')(conv_3)


    pool = MaxPooling1D(pool_size=4)(conv_4)

    #flat = Flatten()(pool)


    dense = Dense(64, kernel_initializer=initializer, activation=activation, kernel_regularizer=regularizer)(
        pool)  # do we need tanh activation here? Ensemble with none mb



    dense = Dense(32, kernel_initializer=initializer, activation=activation, kernel_regularizer=regularizer)(
        dense)  # do we need tanh activation here? Ensemble with none mb



    output = tf.keras.layers.Dense(5,activation='linear',kernel_regularizer=regularizer)(dense)


    lstm_model = tf.keras.Model(inputs=input, outputs=output)



    optimizer = tf.keras.optimizers.Adam(learning_rate=network_args.network['lr'],amsgrad=True)

    lstm_model.compile(
        loss=ohlcv_combined, optimizer=optimizer, metrics=[metric_signs_close,ohlcv_cosine_similarity,ohlcv_mse])

    return lstm_model
