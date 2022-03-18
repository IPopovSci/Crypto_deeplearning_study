
import tensorflow as tf

from tensorflow.keras.layers import Input, GaussianNoise,BatchNormalization, LSTM,LayerNormalization,Dense,MaxPooling1D,Flatten
from keras.layers.convolutional_recurrent import ConvLSTM1D

from pipeline.pipelineargs import PipelineArgs
from Networks.network_config import NetworkParams
from Networks.losses_metrics import ohlcv_mse,ohlcv_cosine_similarity,metric_signs_close,ohlcv_combined

pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()


def convlstm_model():
    batch_size = pipeline_args.args['batch_size']
    time_steps = pipeline_args.args['time_steps']
    num_features = pipeline_args.args['num_features']
    extra_dim = 1


    #input = Input(batch_shape=(batch_size, time_steps, num_features,1))
    input = Input(shape=(time_steps, num_features, 1), batch_size=batch_size)

    regularizer = tf.keras.regularizers.l1_l2(l1=network_args.network['l1_reg'],l2=network_args.network['l2_reg'])
    initializer = tf.keras.initializers.LecunNormal()
    dropout = network_args.network['dropout']

    activation = tf.keras.activations.swish

    x = ConvLSTM1D(64,stateful=False,kernel_size=3,bias_initializer=initializer,bias_regularizer=regularizer,activity_regularizer=regularizer,recurrent_regularizer=regularizer,recurrent_initializer=initializer,activation=activation,kernel_initializer=initializer,kernel_regularizer=regularizer,return_sequences=True,padding='same')(input)

    x = LayerNormalization()(x)

    x = ConvLSTM1D(64,stateful=False,kernel_size=3,bias_initializer=initializer,bias_regularizer=regularizer,activity_regularizer=regularizer,recurrent_regularizer=regularizer,recurrent_initializer=initializer, activation=activation,kernel_initializer=initializer, kernel_regularizer=regularizer,return_sequences=True,padding='same')(x)

    x = LayerNormalization()(x)

    x = ConvLSTM1D(64,stateful=False,kernel_size=3,bias_initializer=initializer,bias_regularizer=regularizer,activity_regularizer=regularizer,recurrent_regularizer=regularizer,recurrent_initializer=initializer, activation=activation,kernel_initializer=initializer, kernel_regularizer=regularizer,return_sequences=False,padding='same')(x)

    x = LayerNormalization()(x)

    x = MaxPooling1D(pool_size=4,activity_regularizer=regularizer)(x)

    x = Flatten()(x)

    x = LayerNormalization()(x)

    x = Dense(32,activation=activation,activity_regularizer=regularizer,kernel_regularizer=regularizer,bias_regularizer=regularizer,kernel_initializer=initializer,bias_initializer=initializer)(
        x)  # do we need tanh activation here? Ensemble with none mb


    x = LayerNormalization()(x)

    x = Dense(16,activation=activation,activity_regularizer=regularizer,kernel_regularizer=regularizer,bias_regularizer=regularizer,kernel_initializer=initializer,bias_initializer=initializer)(
        x)  # do we need tanh activation here? Ensemble with none mb

    x = LayerNormalization()(x)

    output = tf.keras.layers.Dense(5, activation=activation,activity_regularizer=regularizer,kernel_regularizer=regularizer,bias_regularizer=regularizer,kernel_initializer=initializer,bias_initializer=initializer)(x)


    lstm_model = tf.keras.Model(inputs=input, outputs=output)



    optimizer = tf.keras.optimizers.Adam(learning_rate=network_args.network['lr'],amsgrad=True)

    lstm_model.compile(
        loss=ohlcv_combined, optimizer=optimizer, metrics=[metric_signs_close,ohlcv_cosine_similarity,ohlcv_mse])

    return lstm_model
