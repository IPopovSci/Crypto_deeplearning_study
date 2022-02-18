from tensorflow.keras.layers import Dense, Input, Concatenate, Conv1D, MaxPooling1D,Flatten
from Arguments import args
from LSTM.callbacks import custom_loss,ratio_loss,my_metric_fn
from tensorflow.keras.models import load_model
from attention import Attention
import os
import tensorflow as tf
from pipeline import data_prep
from Arguments import args
from Data_Processing.data_trim import trim_dataset
from LSTM.callbacks import mean_squared_error_custom,custom_cosine_similarity,metric_signs,custom_mean_absolute_error,stock_loss,stock_loss_metric
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import os
from keras_self_attention import SeqSelfAttention
import tensorflow as tf
from keras_multi_head import MultiHead,MultiHeadAttention
from Backtesting.Backtesting import correct_signs
import joblib
import numpy as np
import tensorflow.keras.backend as K


def create_model_ensembly(x_t):
    BATCH_SIZE = args['batch_size']
    TIME_STEPS = args['time_steps']
    models = []
    i=0
    for model in os.listdir(f'F:\MM\production\pancake_predictions\models\\1min\\'):
        saved_model = load_model(os.path.join(f'F:\MM\production\pancake_predictions\models\\1min\\', model),
                                 custom_objects={'MultiHead':MultiHead,'stock_loss_metric':stock_loss_metric,'stock_loss':stock_loss,'custom_mean_absolute_error':custom_mean_absolute_error,'metric_signs':metric_signs,'SeqSelfAttention': SeqSelfAttention,'custom_cosine_similarity':custom_cosine_similarity,'mean_squared_error_custom':mean_squared_error_custom})
        saved_model._name = f'Model_{i}'
        i += 1
        saved_model.trainable = False
        models.append(saved_model)

    activation = 'selu'

    regularizer = None#tf.keras.regularizers.l1_l2(l1=0.001,l2=0.001)
    kernel_init = tf.keras.initializers.LecunNormal()

    model_input = Input(batch_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]),name='input_models')

    outputs_models = [model(model_input,training=False) for model in models]

    conv = Conv1D(kernel_size=4,filters=128,kernel_initializer=kernel_init,kernel_regularizer=regularizer,activation=activation)(model_input)

    conv_2 = Conv1D(kernel_size=4,filters=64,kernel_initializer=kernel_init,kernel_regularizer=regularizer,activation=activation)(conv)

    conv_3 = Conv1D(kernel_size=4, filters=32, kernel_initializer=kernel_init, kernel_regularizer=regularizer,
                    activation=activation)(conv_2)

    pool = MaxPooling1D(pool_size=3)(conv_3)

    flat = Flatten()(pool)


    concat = Concatenate()([outputs_models[0],outputs_models[1],outputs_models[2],outputs_models[3],flat])


    Dense_one = Dense(100,kernel_initializer=kernel_init,kernel_regularizer=regularizer,activation=activation)(concat)

    Dense_two = Dense(50,kernel_initializer=kernel_init,kernel_regularizer=regularizer,activation=activation)(Dense_one)

    Dense_two = Dense(25, kernel_initializer=kernel_init, kernel_regularizer=regularizer, activation=activation)(
        Dense_two)

    ensemble_output = Dense(1,kernel_initializer=kernel_init,kernel_regularizer=regularizer,activation=activation)(Dense_two)

    ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output,name=f'ensemble_model')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
    ensemble_model.compile(loss=custom_cosine_similarity, optimizer=optimizer, metrics=metric_signs)

    return ensemble_model



