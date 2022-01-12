from keras.layers import Input,ConvLSTM2D,BatchNormalization,Conv3D
from Arguments import args
import tensorflow as tf


def LSTM_network_conv_create(x_t):
    time_steps = args['time_steps']
    batch_size = args['batch_size']
    # Construct the input layer with no definite frame size.
    inp = Input(batch_shape=(x_t.shape[0],x_t.shape[1],x_t.shape[2],x_t.shape[3],x_t.shape[4]))

    # We will construct 3 `ConvLSTM2D` layers with batch normalization,
    # followed by a `Conv3D` layer for the spatiotemporal outputs.
    x = ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",stateful=True,data_format='channels_last',name='CONVLSTM_1',
    )(inp)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(
        filters=64,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=True,
        activation="relu",
    )(x)
    x = Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
    )(x)

    # Next, we will build the complete model and compile it.
    model = tf.keras.models.Model(inp, x)
    model.compile(
        loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.Adam(),
    )
    return model