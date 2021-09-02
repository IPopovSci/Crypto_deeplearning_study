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


    LSTM_1 = LSTM(int(66), return_sequences=True, stateful=True, kernel_regularizer=regularizer,
             recurrent_dropout=0.3, dropout=0.3, bias_regularizer=tf.keras.regularizers.l2(1e-4),
             activity_regularizer=tf.keras.regularizers.l2(1e-5), activation='elu')(input)

    LSTM_2 = LSTM(int(44), return_sequences=True, stateful=True, kernel_regularizer=regularizer,
                  recurrent_dropout=0.3, dropout=0.3, bias_regularizer=tf.keras.regularizers.l2(1e-4),
                  activity_regularizer=tf.keras.regularizers.l2(1e-5), activation='elu')(LSTM_1)

    LSTM_3 = LSTM(int(66), return_sequences=True, stateful=True, kernel_regularizer=regularizer,
             recurrent_dropout=0.3, dropout=0.3, bias_regularizer=tf.keras.regularizers.l2(1e-4),
             activity_regularizer=tf.keras.regularizers.l2(1e-5), activation='elu')(input)

    LSTM_4 = LSTM(int(44), return_sequences=True, stateful=False, kernel_regularizer=regularizer,
                  recurrent_dropout=0.3, dropout=0.3, bias_regularizer=tf.keras.regularizers.l2(1e-4),
                  activity_regularizer=tf.keras.regularizers.l2(1e-5), activation='elu')(LSTM_3)

    GRU_1 = tf.keras.layers.GRU(66,reset_after=False,dropout=0.3,recurrent_dropout=0.3,return_sequences=True, activation='elu')(input)

    GRU_2 = tf.keras.layers.GRU(44, reset_after=False, dropout=0.3, recurrent_dropout=0.3,return_sequences=True, activation='elu')(GRU_1)

    LSTM_5 = LSTM(int(66), return_sequences=True, stateful=True, kernel_regularizer=regularizer,
             recurrent_dropout=0.3, dropout=0.3, bias_regularizer=tf.keras.regularizers.l2(1e-4),
             activity_regularizer=tf.keras.regularizers.l2(1e-5), activation='elu')(input)

    LSTM_6 = LSTM(int(44), return_sequences=False, stateful=True, kernel_regularizer=regularizer,
                  recurrent_dropout=0.3, dropout=0.3, bias_regularizer=tf.keras.regularizers.l2(1e-4),
                  activity_regularizer=tf.keras.regularizers.l2(1e-5), activation='elu')(LSTM_5)

    LSTM_7 = LSTM(int(66), return_sequences=True, stateful=True, kernel_regularizer=regularizer,
             recurrent_dropout=0.3, dropout=0.3, bias_regularizer=tf.keras.regularizers.l2(1e-4),
             activity_regularizer=tf.keras.regularizers.l2(1e-5), activation='elu')(input)

    LSTM_8 = LSTM(int(44), return_sequences=False, stateful=False, kernel_regularizer=regularizer,
                  recurrent_dropout=0.3, dropout=0.3, bias_regularizer=tf.keras.regularizers.l2(1e-4),
                  activity_regularizer=tf.keras.regularizers.l2(1e-5), activation='elu')(LSTM_7)

    GRU_3 = tf.keras.layers.GRU(66,reset_after=False,dropout=0.3,recurrent_dropout=0.3,return_sequences=True, activation='elu')(input)

    GRU_4 = tf.keras.layers.GRU(44, reset_after=False, dropout=0.3, recurrent_dropout=0.3,return_sequences=False, activation='elu')(GRU_3)

    concat = tf.keras.layers.concatenate([LSTM_2,LSTM_4,GRU_2])

    attention_1 = Attention()(concat)

    concat_2 = tf.keras.layers.concatenate([LSTM_6,LSTM_8,GRU_4,attention_1])

    Dense_4 = tf.keras.layers.Dense(int(88), activation='elu')(concat_2)

    Dense_5 = tf.keras.layers.Dense(int(44), activation='elu')(Dense_4)

    Dense_6 = tf.keras.layers.Dense(int(22), activation='elu')(Dense_5)

    Dense_7 = tf.keras.layers.Dense(int(11), activation='elu')(Dense_6)

    output = tf.keras.layers.Dense(1, activation='elu')(Dense_7)

    lstm_model = tf.keras.Model(inputs=input, outputs=output)
    # lstm_model = CRFModel(lstm_model, 5)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,amsgrad=True,epsilon=0.1)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=0.001) #make it SGD
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=1.,rho= 0.9)
    lstm_model.compile(loss=ratio_loss, optimizer=optimizer,metrics=my_metric_fn)
    return lstm_model
