import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
import os
tf.config.experimental_run_functions_eagerly(True)
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
    alpha = 2000
    direction_loss = tf.compat.v1.scatter_nd_update(direction_loss, indices, alpha * updates)

    custom_loss = K.mean(tf.multiply(K.square((K.log(y_true + 1.)) - (K.log(y_pred + 1.))), direction_loss), axis=-1)

    return custom_loss

def custom_loss_hinge(y_true, y_pred):
    # extract the "next day's price" of tensor
    y_true_next = y_true[1:]
    y_pred_next = y_pred[1:]

    # extract the "today's price" of tensor
    y_true_tdy = y_true[:-1]
    y_pred_tdy = y_pred[:-1]



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
    alpha = 1250

    stability = 0.0000000000001
    direction_loss = tf.compat.v1.scatter_nd_update(direction_loss, indices, alpha * updates)
    percent_loss = K.abs(y_true - y_pred)

    directional_mult = direction_loss

    yukawa_loss = tf.multiply(K.exp(abs(y_true-y_pred + stability))/(y_true+stability),directional_mult)

    directional_mult= tf.multiply(yukawa_loss,tf.multiply(percent_loss,-1))


    custom_loss = K.mean(directional_mult)

    return custom_loss


mcp = ModelCheckpoint(os.path.join('data\output', "best_lstm_model.h5"), monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=False, mode='max', period=1)

# def stock_loss(y_true, y_pred):
#     alpha = 100
#     stability = 0.001
#     true_move = y_true[-1] - y_true[-2]
#     pred_move = y_pred[-1] - y_true[-2]
#     loss = K.switch(K.less(tf.multiply(K.sign(true_move),K.sign(pred_move)),0), #sign of true values, sign of pred values
#         K.square(y_true-y_pred)**(K.abs(y_true**2-y_pred*y_true)+1),
#         K.square(y_true-y_pred)
#         )
#     return K.mean(loss, axis=-1)

def stock_loss(y_true, y_pred):
    loss = K.switch(K.less_equal(tf.multiply(K.sign(y_true),K.sign(y_pred)),0), #sign of true values, sign of pred values
        K.abs(y_true-y_pred)*(K.abs(y_true**2-y_pred*y_true)+10.),
        K.abs(y_true-y_pred)*(K.abs(y_true**2-y_pred*y_true)))
    return K.mean(loss, axis=-1)



def stock_loss_money(y_true, y_pred): #If using this, important to shift data for training
    money_now = 1.
    money_now = K.switch(K.less_equal(tf.multiply(K.sign(y_true),K.sign(y_pred)),0), #sign of true values, sign of pred values
        money_now - y_true,
        money_now + y_true)
    #print('money_now - y_true',y_true.shape,'money_now + y_true',tf.multiply(K.sign(y_true),K.sign(y_pred)).shape,'shape of money:', tf.multiply(K.sign(y_true),K.sign(y_pred)).shape)
    # print('Money:',money_now)
    loss = K.abs(y_true-y_pred) * (1/money_now)
    return K.mean(loss)