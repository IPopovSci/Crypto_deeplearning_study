from tensorflow.keras.layers import LSTM, Dense, Dropout, Input,Activation
from Arguments import args
from callbacks import mcp, custom_loss,custom_loss_hinge,stock_loss,stock_loss_money
from tensorflow.keras.models import Sequential, load_model
from attention import Attention
import os
import tensorflow as tf

def create_model_ensembly(x_t,model_name_load):
    BATCH_SIZE = args['batch_size']
    TIME_STEPS = args['time_steps']
    models = []
    inputs_2 = []
    inputs_3 = []
    inputs_4 = []
    i=0
    j=0
    k=0
    for model in os.listdir(f'data\output\models\{model_name_load}'):
        saved_model = load_model(os.path.join(f'data\output\models\{model_name_load}', model),
                                 custom_objects={'stock_loss':stock_loss,'custom_loss':custom_loss,'custom_loss_hinge': custom_loss_hinge, 'attention': Attention})
        saved_model._name = f'Model_{i}'
        i += 1
        models.append(saved_model)

    model_input = Input(batch_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]),name='input_models')


    for model in models:
        model_outputs = model(model_input)
        model_outputs._name=f'Output1_{j}'
        j+=1
        inputs_2.append(model_outputs)


    for model in inputs_2:
        Dense_l = Dense(10,name=f'Dense_1{k}')(model)
        k+=1
        inputs_4.append(Dense_l)

    concat = tf.keras.layers.concatenate(inputs=inputs_4,axis=-1)

    Dense_one = Dense(25)(concat)

    Dense_two = Dense(10)(Dense_one)

    ensemble_output = Dense(1)(Dense_two)

    ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output,name=f'ensemble_model')
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.01)
    ensemble_model.compile(loss=custom_loss, optimizer=optimizer)

    return ensemble_model

def create_model_ensembly_average(x_t,model_name_load):
    BATCH_SIZE = args['batch_size']
    TIME_STEPS = args['time_steps']
    models = []
    input_1 = []
    i=0
    for model in os.listdir(f'data\output\models\{model_name_load}'):
        saved_model = load_model(os.path.join(f'data\output\models\{model_name_load}', model),
                                 custom_objects={'stock_loss_money':stock_loss_money,'stock_loss':stock_loss,'custom_loss':custom_loss,'custom_loss_hinge': custom_loss_hinge, 'attention': Attention})
        saved_model._name = f'Model_{i}'
        i += 1
        models.append(saved_model)

    model_input = Input(batch_shape=(BATCH_SIZE, TIME_STEPS, x_t.shape[2]),name='input_models')

    outputs = [model(model_input) for model in models]

    ensemble_output = tf.keras.layers.Average()(outputs)

    ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output,name=f'ensemble_model_average')
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.0001, rho=0.9)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.000001)
    ensemble_model.compile(loss=custom_loss, optimizer=optimizer)

    return ensemble_model


