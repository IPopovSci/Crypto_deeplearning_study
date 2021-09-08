from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from Arguments import args
from callbacks import custom_loss,ratio_loss,my_metric_fn
import tensorflow as tf
from attention import Attention
from keras_crf import CRFModel




def create_lstm_model(x_t):
    BATCH_SIZE = args['batch_size']
    TIME_STEPS = args['time_steps']
    n_components = args['n_components']

    input = Input(batch_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]))
    regularizer = tf.keras.regularizers.l1_l2(1e-4)


    LSTM_1 = LSTM(int(44), return_sequences=True, stateful=True, kernel_regularizer=regularizer,
             recurrent_dropout=0.3, dropout=0.3, bias_regularizer=tf.keras.regularizers.l2(1e-4),
             activity_regularizer=tf.keras.regularizers.l2(1e-5))(input)

    LSTM_2 = LSTM(int(22), return_sequences=True, stateful=True, kernel_regularizer=regularizer,
                  recurrent_dropout=0.3, dropout=0.3, bias_regularizer=tf.keras.regularizers.l2(1e-4),
                  activity_regularizer=tf.keras.regularizers.l2(1e-5))(LSTM_1)

    LSTM_10 = LSTM(int(11), return_sequences=True, stateful=True, kernel_regularizer=regularizer,
             recurrent_dropout=0.3, dropout=0.3, bias_regularizer=tf.keras.regularizers.l2(1e-4),
             activity_regularizer=tf.keras.regularizers.l2(1e-5))(LSTM_2)

    # LSTM_20 = LSTM(int(1), return_sequences=True, stateful=True, kernel_regularizer=regularizer,
    #               recurrent_dropout=0.3, dropout=0.3, bias_regularizer=tf.keras.regularizers.l2(1e-4),
    #               activity_regularizer=tf.keras.regularizers.l2(1e-5))(LSTM_10)

    # LSTM_3 = LSTM(int(44), return_sequences=True, stateful=True, kernel_regularizer=regularizer,
    #          recurrent_dropout=0.3, dropout=0.3, bias_regularizer=tf.keras.regularizers.l2(1e-4),
    #          activity_regularizer=tf.keras.regularizers.l2(1e-5))(input)
    #
    # LSTM_4 = LSTM(int(22), return_sequences=True, stateful=False, kernel_regularizer=regularizer,
    #               recurrent_dropout=0.3, dropout=0.3, bias_regularizer=tf.keras.regularizers.l2(1e-4),
    #               activity_regularizer=tf.keras.regularizers.l2(1e-5))(LSTM_3)

    GRU_1 = tf.keras.layers.GRU(44,reset_after=False,dropout=0.3,recurrent_dropout=0.3,return_sequences=True)(input)

    GRU_2 = tf.keras.layers.GRU(22, reset_after=False, dropout=0.3, recurrent_dropout=0.3,return_sequences=True)(GRU_1)

    GRU_10 = tf.keras.layers.GRU(11,reset_after=False,dropout=0.3,recurrent_dropout=0.3,return_sequences=True)(GRU_2)
    #
    # GRU_20 = tf.keras.layers.GRU(1, reset_after=False, dropout=0.3, recurrent_dropout=0.3,return_sequences=True)(GRU_10)


    LSTM_5 = LSTM(int(44), return_sequences=True, stateful=True, kernel_regularizer=regularizer,
             recurrent_dropout=0.3, dropout=0.3, bias_regularizer=tf.keras.regularizers.l2(1e-4),
             activity_regularizer=tf.keras.regularizers.l2(1e-5))(input)

    LSTM_6 = LSTM(int(22), return_sequences=True, stateful=True, kernel_regularizer=regularizer,
                  recurrent_dropout=0.3, dropout=0.3, bias_regularizer=tf.keras.regularizers.l2(1e-4),
                  activity_regularizer=tf.keras.regularizers.l2(1e-5))(LSTM_5)

    LSTM_50 = LSTM(int(11), return_sequences=False, stateful=True, kernel_regularizer=regularizer,
             recurrent_dropout=0.3, dropout=0.3, bias_regularizer=tf.keras.regularizers.l2(1e-4),
             activity_regularizer=tf.keras.regularizers.l2(1e-5))(LSTM_6)

    # LSTM_60 = LSTM(int(1), return_sequences=False, stateful=True, kernel_regularizer=regularizer,
    #               recurrent_dropout=0.3, dropout=0.3, bias_regularizer=tf.keras.regularizers.l2(1e-4),
    #               activity_regularizer=tf.keras.regularizers.l2(1e-5))(LSTM_50)
    #
    # # LSTM_7 = LSTM(int(44), return_sequences=True, stateful=True, kernel_regularizer=regularizer,
    # #          recurrent_dropout=0.3, dropout=0.3, bias_regularizer=tf.keras.regularizers.l2(1e-4),
    # #          activity_regularizer=tf.keras.regularizers.l2(1e-5))(input)
    # #
    # # LSTM_8 = LSTM(int(22), return_sequences=False, stateful=False, kernel_regularizer=regularizer,
    # #               recurrent_dropout=0.3, dropout=0.3, bias_regularizer=tf.keras.regularizers.l2(1e-4),
    # #               activity_regularizer=tf.keras.regularizers.l2(1e-5))(LSTM_7)
    #
    GRU_3 = tf.keras.layers.GRU(44,reset_after=False,dropout=0.3,recurrent_dropout=0.3,return_sequences=True)(input)

    GRU_4 = tf.keras.layers.GRU(22, reset_after=False, dropout=0.3, recurrent_dropout=0.3,return_sequences=True)(GRU_3)

    GRU_30 = tf.keras.layers.GRU(11,reset_after=False,dropout=0.3,recurrent_dropout=0.3,return_sequences=False)(GRU_4)
    #
    # GRU_40 = tf.keras.layers.GRU(1, reset_after=False, dropout=0.3, recurrent_dropout=0.3,return_sequences=False)(GRU_30)

    concat = tf.keras.layers.concatenate([LSTM_10,GRU_10])

    attention_1 = Attention()(concat)
    #
    concat_2 = tf.keras.layers.concatenate([LSTM_50,GRU_30,attention_1])

    Dense_4 = tf.keras.layers.Dense(100,activation='sigmoid')(concat_2)

    output = tf.keras.layers.Dense(1, activation='sigmoid')(Dense_4)

    lstm_model = tf.keras.Model(inputs=input, outputs=output)
    # lstm_model = CRFModel(lstm_model, 5)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,amsgrad=True,epsilon=1.)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=0.000000001,nesterov=True,momentum=0.3) #make it SGD
    #optimizer = tf.keras.optimizers.Adadelta(learning_rate=1,rho= 0.95)
    lstm_model.compile(loss=ratio_loss, optimizer=optimizer,metrics=my_metric_fn)
    return lstm_model
