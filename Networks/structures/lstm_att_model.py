import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, GaussianNoise, LSTM, TimeDistributed, LayerNormalization,BatchNormalization

from keras_self_attention import SeqSelfAttention
from pipeline.pipelineargs import PipelineArgs
from Networks.network_config import NetworkParams
from Networks.losses_metrics import ohlcv_mse, ohlcv_cosine_similarity, metric_signs_close, ohlcv_combined

pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()

'''This one is based on the article about time series and self attention'''


def lstm_att_model():
    batch_size = pipeline_args.args['batch_size']
    time_steps = pipeline_args.args['time_steps']
    num_features = pipeline_args.args['num_features']

    input = Input(batch_shape=(batch_size, time_steps, num_features))

    regularizer = tf.keras.regularizers.l1_l2(l1=network_args.network['l1_reg'], l2=network_args.network['l2_reg'])
    initializer = tf.keras.initializers.LecunNormal()
    dropout = network_args.network['dropout']
    bias_initializer = initializer

    activation = tf.keras.activations.swish

    # #This is First side-chain: input>LSTM(stateful)>LSTM(stateful)>TD Dense layer. The output is a 3d vector
    LSTM_1 = LSTM(int(75), return_sequences=True, stateful=True, activation=activation, kernel_initializer=initializer,
                  bias_initializer=bias_initializer,recurrent_initializer=initializer,bias_regularizer=regularizer,kernel_regularizer=regularizer)(input)

    x = LayerNormalization()(LSTM_1)

    Dense_1 = Dense(50, activation=activation, kernel_initializer=initializer, bias_initializer=bias_initializer,bias_regularizer=regularizer,kernel_regularizer=regularizer)(x)

    # This is the attention side-chain: LSTM(Stateless)>LSTM>Attention. The output is a 3d vector

    LSTM_3 = LSTM(int(75), return_sequences=True, stateful=False, activation=activation, kernel_initializer=initializer,
                  bias_initializer=bias_initializer,recurrent_initializer=initializer,bias_regularizer=regularizer,kernel_regularizer=regularizer)(input)

    x = LayerNormalization()(LSTM_3)

    attention_1 = SeqSelfAttention(units=50,bias_initializer=bias_initializer,bias_regularizer=regularizer,kernel_regularizer=regularizer)(x)

    concat = tf.keras.layers.concatenate([Dense_1, attention_1])

    x = BatchNormalization()(concat)

    Dense_fin = Dense(125, activation=activation, kernel_initializer=initializer, bias_initializer=bias_initializer,bias_regularizer=regularizer,kernel_regularizer=regularizer)(
        x)

    x = BatchNormalization()(Dense_fin)

    Dense_fin_2 = Dense(75, activation=activation, kernel_initializer=initializer, bias_initializer=bias_initializer,bias_regularizer=regularizer,kernel_regularizer=regularizer)(
        x)

    x = BatchNormalization()(Dense_fin_2)

    output = tf.keras.layers.Dense(5, activation='linear', kernel_initializer=initializer, kernel_regularizer=regularizer, bias_initializer=bias_initializer,bias_regularizer=regularizer)(x)

    lstm_model = tf.keras.Model(inputs=input, outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=network_args.network['lr'], amsgrad=True)

    lstm_model.compile(
        loss=ohlcv_cosine_similarity, optimizer=optimizer, metrics=[metric_signs_close, ohlcv_cosine_similarity])

    return lstm_model
