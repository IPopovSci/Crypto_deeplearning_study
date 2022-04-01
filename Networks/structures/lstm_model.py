import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, LSTM,LayerNormalization

from pipeline.pipelineargs import PipelineArgs
from Networks.network_config import NetworkParams
from Networks.losses_metrics import ohlcv_cosine_similarity, metric_signs_close, metric_loss

pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()

def lstm_att_model():
    batch_size = pipeline_args.args['batch_size']
    time_steps = pipeline_args.args['time_steps']
    num_features = pipeline_args.args['num_features']

    input = Input(batch_shape=(batch_size, time_steps, num_features))

    regularizer = None#tf.keras.regularizers.l1_l2(l1=network_args.network['l1_reg'], l2=network_args.network['l2_reg'])
    initializer = tf.keras.initializers.glorot_uniform()
    dropout = network_args.network['dropout']


    activation = tf.keras.activations.swish

    x = LSTM(int(60), return_sequences=True, stateful=False, activation=activation, kernel_initializer=initializer)(input)

    x = LayerNormalization()(x)

    x = LSTM(int(35), return_sequences=True, stateful=False, activation=activation, kernel_initializer=initializer)(x)

    x = LayerNormalization()(x)
    x = LSTM(int(20), return_sequences=True, stateful=False, activation=activation, kernel_initializer=initializer)(x)

    x = LayerNormalization()(x)

    x = LSTM(int(10), return_sequences=True, stateful=False, activation=activation, kernel_initializer=initializer)(x)

    x = LayerNormalization()(x)

    x = Dense(10, activation=activation, kernel_initializer=initializer,
              activity_regularizer=regularizer)(x)

    x = LayerNormalization()(x)

    output = tf.keras.layers.Dense(5, activation='softsign',
                                   kernel_initializer=initializer)(x)

    lstm_model = tf.keras.Model(inputs=input, outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=network_args.network['lr'], amsgrad=True)

    lstm_model.compile(
        loss=profit_ratio_assymetric, optimizer=optimizer, metrics=[metric_signs_close, ohlcv_cosine_similarity, ohlcv_mse,metric_profit_ratio])

    return lstm_model
