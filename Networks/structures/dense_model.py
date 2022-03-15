
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, GaussianNoise,Dropout

from pipeline.pipelineargs import PipelineArgs
from Networks.network_config import NetworkParams
from Networks.losses_metrics import ohlcv_mse,ohlcv_cosine_similarity,metric_signs_close,ohlcv_combined

pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()


def dense_model():
    batch_size = pipeline_args.args['batch_size']
    time_steps = pipeline_args.args['time_steps']
    num_features = pipeline_args.args['num_features']


    input = Input(batch_shape=(batch_size, time_steps, num_features))

    regularizer = tf.keras.regularizers.l1_l2(l1=network_args.network['l1_reg'],l2=network_args.network['l2_reg'])
    initializer = tf.keras.initializers.LecunNormal()
    dropout = network_args.network['dropout']

    activation = 'selu'

    noise = GaussianNoise(0.05)(input)

    x = Dense(160,activation=activation,kernel_regularizer=regularizer,activity_regularizer=regularizer,kernel_initializer=initializer,bias_initializer=initializer)(noise)

    x = Dropout(dropout)(x)

    x = Dense(80,activation=activation,kernel_regularizer=regularizer,activity_regularizer=regularizer,kernel_initializer=initializer,bias_initializer=initializer)(x)

    x = Dropout(dropout)(x)

    x = Dense(40,activation=activation,kernel_regularizer=regularizer,activity_regularizer=regularizer,kernel_initializer=initializer,bias_initializer=initializer)(x)


    output = tf.keras.layers.Dense(5,activation='linear',kernel_regularizer=regularizer)(x)


    lstm_model = tf.keras.Model(inputs=input, outputs=output)



    optimizer = tf.keras.optimizers.Adam(learning_rate=network_args.network['lr'],amsgrad=True)

    lstm_model.compile(
        loss=ohlcv_combined, optimizer=optimizer, metrics=[metric_signs_close,ohlcv_cosine_similarity,ohlcv_mse])

    return lstm_model
