
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, GaussianNoise,BatchNormalization,Conv2D,LSTM
from pipeline import pipelineargs
from keras.layers.convolutional_recurrent import ConvLSTM1D

from pipeline.pipelineargs import PipelineArgs
from Networks.network_config import NetworkParams
from Networks.losses_metrics.losses_metrics import ohlcv_mse,ohlcv_cosine_similarity,metric_signs_close,ohlcv_combined

pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()


def create_convlstm_model():
    batch_size = pipeline_args.args['batch_size']
    time_steps = pipeline_args.args['time_steps']
    num_features = pipeline_args.args['num_features']
    extra_dim = 1


    #input = Input(batch_shape=(batch_size, time_steps, num_features,1))
    input = Input(shape=(time_steps, num_features, 1), batch_size=batch_size)

    regularizer = tf.keras.regularizers.l1_l2(l1=network_args.network['l1_reg'],l2=network_args.network['l2_reg'])
    initializer = tf.keras.initializers.LecunNormal()
    dropout = network_args.network['dropout']

    activation = 'softsign'

    noise = GaussianNoise(0.05)(input)

    convlstm = ConvLSTM1D(64,stateful=True,kernel_size=5,recurrent_initializer=initializer,activation=activation,kernel_initializer=initializer,kernel_regularizer=regularizer,return_sequences=True,padding='same')(noise)

    convlstm = BatchNormalization()(convlstm)

    convlstm = ConvLSTM1D(64,stateful=True,kernel_size=3,recurrent_initializer=initializer, activation=activation,kernel_initializer=initializer, kernel_regularizer=regularizer,return_sequences=True,padding='same')(convlstm)

    convlstm = BatchNormalization()(convlstm)

    convlstm = ConvLSTM1D(64,stateful=True,kernel_size=1,recurrent_initializer=initializer, activation=activation,kernel_initializer=initializer, kernel_regularizer=regularizer,return_sequences=False,padding='same')(convlstm)

    output = LSTM(5,stateful=True,activation='linear',return_sequences=True)(convlstm)

    #output = tf.keras.layers.Dense(5,activation='softsign',kernel_regularizer=regularizer)(output)


    lstm_model = tf.keras.Model(inputs=input, outputs=output)



    optimizer = tf.keras.optimizers.Adam(learning_rate=network_args.network['lr'],amsgrad=True)

    lstm_model.compile(
        loss=ohlcv_combined, optimizer=optimizer, metrics=[metric_signs_close,ohlcv_cosine_similarity,ohlcv_mse])

    return lstm_model
