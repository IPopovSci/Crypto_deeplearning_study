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


    LSTM_1 = LSTM(int(88), return_sequences=True, stateful=True, kernel_regularizer=regularizer,
             recurrent_dropout=0.3, dropout=0.3, bias_regularizer=tf.keras.regularizers.l2(1e-4),
             activity_regularizer=tf.keras.regularizers.l2(1e-5))(input)

    LSTM_2 = LSTM(int(66), return_sequences=False, stateful=True, kernel_regularizer=regularizer,
                  recurrent_dropout=0.3, dropout=0.3, bias_regularizer=tf.keras.regularizers.l2(1e-4),
                  activity_regularizer=tf.keras.regularizers.l2(1e-5))(LSTM_1)

    Dense_1 = Dense(10)(LSTM_2)


    output = tf.keras.layers.Dense(5, activation='sigmoid')(Dense_1)

    lstm_model = tf.keras.Model(inputs=input, outputs=output)
    # lstm_model = CRFModel(lstm_model, 5)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=0.000000001,nesterov=True,momentum=0.3) #make it SGD
    #optimizer = tf.keras.optimizers.Adadelta(learning_rate=1,rho= 0.95)
    lstm_model.compile(loss=custom_loss, optimizer=optimizer,metrics=my_metric_fn)
    return lstm_model
