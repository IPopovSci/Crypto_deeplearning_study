from tensorflow.keras.layers import Dense, Input, Concatenate
from pipeline_args import args
from Old_and_crap.callbacks import custom_cosine_similarity,metric_signs,custom_mean_absolute_error
from tensorflow.keras.models import load_model
import os
from keras_self_attention import SeqSelfAttention
import tensorflow as tf
from keras_multi_head import MultiHead


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



    concat = Concatenate()([outputs_models[0],outputs_models[1],outputs_models[2],outputs_models[3],outputs_models[4],outputs_models[5],outputs_models[6],outputs_models[7]])


    Dense_one = Dense(100,kernel_initializer=kernel_init,kernel_regularizer=regularizer,activation=activation)(concat)

    Dense_two = Dense(50,kernel_initializer=kernel_init,kernel_regularizer=regularizer,activation=activation)(Dense_one)

    Dense_two = Dense(25, kernel_initializer=kernel_init, kernel_regularizer=regularizer, activation=activation)(
        Dense_two)

    ensemble_output = Dense(1,kernel_initializer=kernel_init,kernel_regularizer=regularizer,activation=activation)(Dense_two)

    ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output,name=f'ensemble_model')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    ensemble_model.compile(loss=[custom_cosine_similarity,'mse'], optimizer=optimizer, metrics=metric_signs)

    return ensemble_model



