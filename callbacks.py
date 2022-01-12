import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import numpy as np
from Arguments import args
tf.config.experimental_run_functions_eagerly(True)
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

#TODO: Custom loss that simulates buying - add when moving in the same direction, substract when different
def custom_loss(y_true, y_pred):
    # extract the "next day's price" of tensor
    y_true_next = y_true[1:]
    y_pred_next = y_pred[1:]

    # extract the "today's price" of tensor
    y_true_tdy = y_true[:-1]
    y_pred_tdy = y_pred[:-1]

    #print('Shape of y_pred_back -', y_pred_tdy.get_shape())

    # substract to get up/down movement of the two tensors
    y_true_diff = tf.subtract(y_true_next, y_true_tdy)
    y_pred_diff = tf.subtract(y_pred_next, y_pred_tdy)


    # create a standard tensor with zero value for comparison
    standard = tf.zeros_like(y_pred_diff)

    # compare with the standard; if true, UP; else DOWN
    y_true_move = tf.greater_equal(y_true_diff, standard)
    y_pred_move = tf.greater_equal(y_pred_diff, standard)
    y_true_move = tf.reshape(y_true_move, [-1])
    y_pred_move = tf.reshape(y_pred_move, [-1])

    # find indices where the directions are not the same
    condition = tf.not_equal(y_true_move, y_pred_move)
    indices = tf.where(condition)

    # move one position later
    ones = tf.ones_like(indices)
    indices = tf.add(indices, ones)
    indices = K.cast(indices, dtype='int32')

    # create a tensor to store directional loss and put it into custom loss output
    direction_loss = tf.Variable(tf.ones_like(y_pred), dtype='float32')
    updates = K.cast(tf.ones_like(indices), dtype='float32')
    alpha = 2500
    direction_loss = tf.compat.v1.scatter_nd_update(direction_loss, indices, alpha * updates)

    custom_loss = K.mean(tf.multiply(K.square((K.log(y_true + 1.)) - (K.log(y_pred + 1.))), direction_loss), axis=-1)

    return custom_loss

def custom_loss_direction(y_true, y_pred):
    # extract the "next day's price" of tensor
    y_true_next = y_true[1:]
    y_pred_next = y_pred[1:]

    # extract the "today's price" of tensor
    y_true_tdy = y_true[:-1]
    y_pred_tdy = y_pred[:-1]

    #print('Shape of y_pred_back -', y_pred_tdy.get_shape())

    # substract to get up/down movement of the two tensors
    y_true_diff = tf.subtract(y_true_next, y_true_tdy)
    y_pred_diff = tf.subtract(y_pred_next, y_pred_tdy)


    # create a standard tensor with zero value for comparison
    standard = tf.zeros_like(y_pred_diff)

    # compare with the standard; if true, UP; else DOWN
    y_true_move = tf.greater_equal(y_true_diff, standard)
    y_pred_move = tf.greater_equal(y_pred_diff, standard)
    y_true_move = tf.reshape(y_true_move, [-1])
    y_pred_move = tf.reshape(y_pred_move, [-1])

    # find indices where the directions are not the same
    condition = tf.not_equal(y_true_move, y_pred_move)
    indices = tf.where(condition)

    # move one position later
    ones = tf.ones_like(indices)
    indices = tf.add(indices, ones)
    indices = K.cast(indices, dtype='int32')

    # create a tensor to store directional loss and put it into custom loss output
    direction_loss = tf.Variable(tf.ones_like(y_pred), dtype='float32')
    updates = K.cast(tf.ones_like(indices), dtype='float32')
    alpha = 10000
    direction_loss = tf.compat.v1.scatter_nd_update(direction_loss, indices, alpha * updates)

    custom_loss = direction_loss

    return custom_loss

def ratio_loss(y_true, y_pred):
    batch_size = args['batch_size']
    i = 0
    true_sign_list = []
    soft_sign_error = []
    mse = []

    while i > -len(y_pred) + 1:
        prediction = float((y_pred[i - 1] - y_pred[i - 2]) / (y_pred[i - 2]+0.000000001))
        #print(prediction)
        true = float((y_true[i] - y_true[i - 1]) / (y_true[i - 1]+0.000000001))

        square_error_f = tf.abs(tf.divide(tf.subtract(tf.square(true), tf.multiply(prediction, true)),
                                 tf.square(true) + 0.00000000001))

        # x = tf.subtract(tf.add(tf.abs(prediction),tf.abs(true)),tf.abs(prediction+true))

        # z = tf.multiply(y,x)

        # tf.cond(prediction[i]==0,true_fn=soft_sign_error.append(y[0]*100),None)
        #mse.append(tf.sqrt(tf.abs(true-prediction)))
        if prediction == float(0):
            true_sign_list.append(2)
        elif tf.sign(prediction) != tf.sign(true):
            true_sign_list.append(3)
        else:
            true_sign_list.append(1)
        #soft_sign_error.append(y / (tf.abs(y) + 1)) #This should indicate how far away we are from 0 - which is when the signs of preds and truth allign. However x isn't a very good formula here
        i -= 1

    y_true_tdy = y_true[1:]
    y_pred_for_tdy = y_pred[:-1]
    y_true_next_1 = y_true[1:]
    y_pred_next_1 = y_pred[1:]

    # extract the "today's price" of tensor
    y_true_tdy_1 = y_true[:-1]
    y_pred_tdy_1 = y_pred[:-1]
    #square_error = tf.divide(tf.subtract(tf.square(y_true_tdy), tf.multiply(y_true_tdy,y_pred_for_tdy)),tf.square(y_true_tdy)+0.00000000001)
    square_error = tf.abs(tf.subtract(y_true_tdy, y_pred_for_tdy))
    soft_sign_error = square_error

    mse = tf.abs(((y_true_next_1 - y_true_tdy_1) - (y_pred_next_1 - y_pred_for_tdy)) / ((y_pred_next_1 - y_pred_for_tdy)+0.000000001))
    #mse = K.switch(((y_pred_next_1-y_pred_for_tdy)==0),(tf.abs((y_true_next_1 - y_true_tdy_1) - (y_pred_next_1-y_pred_for_tdy))*2),(tf.abs((y_true_next_1 - y_true_tdy_1) - (y_pred_next_1-y_pred_for_tdy))))
    true_preds = tf.Variable(true_sign_list,shape=(len(true_sign_list)),dtype='float')

    #mse = tf.Variable(mse,shape=(len(mse),1),dtype='float')

    return tf.reduce_mean(true_preds) * tf.reduce_mean(soft_sign_error)#*custom_loss_direction(y_true,y_pred)) #* custom_loss_direction(y_true,y_pred)#tf.reduce_mean(soft_sign_error) #tf.reduce_mean(true_preds)*tf.reduce_mean(soft_sign_error)
    #for 2 stable models have only tf.reduce_mean(true_preds) * tf.reduce_mean(soft_sign_error) here w/ append of 10 for wrong direction
mcp = ModelCheckpoint(os.path.join('data\output', "best_lstm_model.h5"), monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=False, mode='max', period=1)
#TODO: Debug this, I have a hunch it doesn't work right when calculating the metric
def my_metric_fn(y_true, y_pred):
    i = -1
    prediction_list = []
    true_list = []
    true_sign_list = []
    false_sign_list = []
    while i > -len(y_pred) + 1:
        prediction = (y_pred[i-1] - y_pred[i - 2]) / y_pred[i-2] * 100
        true = (y_true[i] - y_true[i - 1]) / y_true[i-1] * 100
        if prediction == 0:
            true_sign_list.append(0)
        elif abs(prediction) + abs(true) == abs(prediction + true):
            true_sign_list.append(1)
        else:
            true_sign_list.append(-1)
        prediction_list.append(prediction)
        true_list.append(true)

        i -= 1
    true_preds = tf.Variable(true_sign_list,shape=(len(true_sign_list)))
    # print(true_preds)

    return tf.reduce_sum(true_preds)

def mean_squared_error_custom(y_true, y_pred):
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    y_pred_sign = math_ops.sign(y_pred)
    y_true_sign = math_ops.sign(y_true)
    abs_sign = math_ops.abs(math_ops.subtract(y_pred_sign,y_true_sign))
    first_log = math_ops.log(K.maximum(y_pred, K.epsilon()) + 1.)
    second_log = math_ops.log(K.maximum(y_true, K.epsilon()) + 1.)
    loss_sign = math_ops.add(math_ops.multiply(math_ops.abs(math_ops.subtract(0, math_ops.multiply(y_pred,100))),abs_sign),(math_ops.squared_difference(y_pred, y_true)))

    return K.mean(loss_sign) #Substracting by how close it is off
tf.keras.losses.MeanSquaredLogarithmicError()