from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras import optimizers
from Arguments import args
from build_timeseries import build_timeseries
from callbacks import custom_loss

def create_lstm_model(x_t):
    BATCH_SIZE = args['batch_size']
    TIME_STEPS = args['time_steps']

    lstm_model = Sequential()
    lstm_model.add(LSTM(100, batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]),
                        dropout=0, recurrent_dropout=0,
                        stateful=True, return_sequences=True,
                        kernel_initializer='random_uniform'))
    #lstm_model.add(Dropout(0.4))

    lstm_model.add(LSTM(60, dropout=0.0, recurrent_dropout = 0.15))
    #lstm_model.add(Dropout(0.4))

    lstm_model.add(Dense(20, activation='relu'))
    lstm_model.add(Dense(1, activation='sigmoid'))

    # compile the model
    optimizer = optimizers.Adam(lr=args["LR"])
    lstm_model.compile(loss=custom_loss, optimizer=optimizer)

    return lstm_model