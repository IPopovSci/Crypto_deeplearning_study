import tensorflow as tf
import tensorflow.keras.backend as K
from dotenv import load_dotenv
from tensorflow.python.ops import math_ops
from pipeline.pipelineargs import PipelineArgs

load_dotenv()

pipeline_args = PipelineArgs.get_instance()

batch_size = pipeline_args.args['batch_size']

'''Custom cosine similarity loss
Extracts the last time step and unscales y values before computing loss
Final result has 1. added to it so that the loss is normalized to 0-2 range'''


def ohlcv_cosine_similarity(y_true, y_pred):
    if pipeline_args.args['expand_dims'] == False:
        y_pred = y_pred[:, -1, :]  # Because Dense predictions will have timesteps

    loss = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-1)

    return loss + 1.0


'''Custom MSE loss
Extracts the last time step and computes MSE loss'''


def ohlcv_mse(y_true, y_pred):
    if pipeline_args.args['expand_dims'] == False:
        y_pred = y_pred[:, -1, :]  # Because Dense predictions will have timesteps

    loss = K.mean(K.square(y_true - y_pred), axis=-1)

    return loss


'''Custom absolute value loss
Extracts the last time step and computes absolute value loss'''


def ohlcv_abs(y_true, y_pred):
    if pipeline_args.args['expand_dims'] == False:
        y_pred = y_pred[:, -1, :]  # Because Dense predictions will have timesteps

    loss = K.mean(K.abs(y_true - y_pred), axis=-1)

    return loss


'''Assymetric loss based on absolute value loss
Checks if the product of signs of true and pred is below 0,
if it is, applies a penalty (alpha).
Otherwise acts as a normal absolute value test'''


def assymetric_loss(y_true, y_pred):
    if pipeline_args.args['expand_dims'] == False:
        y_pred = y_pred[:, -1, :]  # Because Dense predictions will have timesteps

    alpha = 100.
    loss = K.switch(K.less(y_true * y_pred, 0),
                    alpha * y_pred ** 2 + K.abs(y_true) - K.sign(y_true) * y_pred,
                    K.abs(y_true - y_pred)
                    )
    return K.mean(loss, axis=-1)

'''Assymetric loss based on mean squared error loss
Checks if the product of signs of true and pred is below 0,
if it is, applies a penalty (alpha).
Otherwise acts as a normal absolute value test'''
def assymetric_loss_mse(y_true, y_pred):
    if pipeline_args.args['expand_dims'] == False:
        y_pred = y_pred[:, -1, :]  # Because Dense predictions will have timesteps

    alpha = 100.
    loss = K.switch(K.less(y_true * y_pred, 0),
                    alpha * y_pred**2 + K.square(y_true-y_pred),
                    K.square(y_true-y_pred)
                    )
    return K.mean(loss, axis=-1)


'''Combined losses
Multiplication of several losses above'''


def ohlcv_combined(y_true, y_pred):
    loss = (ohlcv_mse(y_true, y_pred) * (ohlcv_cosine_similarity(y_true, y_pred)))

    return loss


def assymetric_combined(y_true, y_pred):
    loss = assymetric_loss(y_true, y_pred) * ohlcv_cosine_similarity(y_true, y_pred)
    return loss


def assymetric_mse_combined(y_true, y_pred):
    loss = assymetric_loss_mse(y_true, y_pred) + ohlcv_cosine_similarity(y_true, y_pred)
    return loss


def metric_loss(y_true, y_pred):
    loss = ohlcv_combined(y_true, y_pred) * assymetric_loss(y_true, y_pred)
    return loss


def metric_profit_ratio(y_true, y_pred):
    if pipeline_args.args['expand_dims'] == False:
        y_pred = y_pred[:, -1, :]

    ratio = tf.math.divide_no_nan(y_pred, y_true)
    loss = K.switch(K.greater_equal(K.abs(ratio), 1.), tf.math.divide_no_nan(1., ratio), ratio)
    return K.mean(-loss + 1., axis=-1)


def profit_ratio_mse(y_true, y_pred):
    loss = metric_profit_ratio(y_true, y_pred) * ohlcv_cosine_similarity(y_true, y_pred) * ohlcv_mse(y_true, y_pred)
    return loss


def profit_ratio_cosine(y_true, y_pred):
    loss = metric_profit_ratio(y_true, y_pred) * ohlcv_cosine_similarity(y_true, y_pred)
    return loss


def profit_ratio_assymetric(y_true, y_pred):
    loss = assymetric_loss_mse(y_true, y_pred)+ metric_profit_ratio(y_true, y_pred) + ohlcv_cosine_similarity(y_true,y_pred) #
    return loss


'''Metric that compares how many signs are correct between true and pred values'''


def metric_signs_close(y_true, y_pred):
    if pipeline_args.args['expand_dims'] == False:
        y_pred = y_pred[:, -1, :]  # Because Dense predictions will have timesteps

    y_true_sign = math_ops.sign(y_true)
    y_pred_sign = math_ops.sign(y_pred)

    metric = math_ops.divide(math_ops.abs(math_ops.subtract(y_true_sign, y_pred_sign)), 2.)

    return math_ops.multiply(
        math_ops.divide(math_ops.subtract(float(pipeline_args.args['batch_size']), K.sum(metric) / 5), float(pipeline_args.args['batch_size'])),
        100.)
