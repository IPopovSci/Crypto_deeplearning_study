import os

import joblib
import tensorflow as tf
import tensorflow.keras.backend as K
from dotenv import load_dotenv
from tensorflow.python.framework import ops
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

    return 100 * (loss + 1.0)


'''Custom MSE loss
Extracts the last time step and unscales y values before computing loss'''


def ohlcv_mse(y_true, y_pred):
    if pipeline_args.args['expand_dims'] == False:
        y_pred = y_pred[:, -1, :]  # Because Dense predictions will have timesteps

    loss = K.mean(K.square(y_true - y_pred), axis=-1)

    return loss


def ohlcv_abs(y_true, y_pred):
    if pipeline_args.args['expand_dims'] == False:
        y_pred = y_pred[:, -1, :]  # Because Dense predictions will have timesteps

    loss = K.mean(K.abs(y_true - y_pred), axis=-1)

    return loss


def assymetric_loss(y_true, y_pred):
    if pipeline_args.args['expand_dims'] == False:
        y_pred = y_pred[:, -1, :]  # Because Dense predictions will have timesteps

    alpha = 100.
    loss = K.switch(K.less(y_true * y_pred, 0),
                    alpha * y_pred ** 2 + K.abs(y_true) - K.sign(y_true) * y_pred,
                    K.abs(y_true - y_pred)
                    )
    return K.mean(loss, axis=-1)


'''Combined cosine similarity and MSE loss
CS loss is squared to put a higher emphasis on the correct direction'''


def ohlcv_combined(y_true, y_pred):
    loss = (ohlcv_mse(y_true, y_pred) * (ohlcv_cosine_similarity(y_true, y_pred)))

    return loss


def assymetric_combined(y_true, y_pred):
    loss = assymetric_loss(y_true, y_pred) * ohlcv_cosine_similarity(y_true, y_pred)
    return loss


def metric_loss(y_true, y_pred):
    loss = ohlcv_combined(y_true, y_pred) * assymetric_loss(y_true, y_pred)
    return loss


def metric_signs_close(y_true, y_pred):
    if pipeline_args.args['expand_dims'] == False:
        y_pred = y_pred[:, -1, :]  # Because Dense predictions will have timesteps

    y_true_sign = math_ops.sign(y_true)
    y_pred_sign = math_ops.sign(y_pred)

    metric = math_ops.divide(math_ops.abs(math_ops.subtract(y_true_sign, y_pred_sign)), 2.)

    return math_ops.multiply(math_ops.divide(math_ops.subtract(float(batch_size), K.sum(metric)/5), float(batch_size)),
                             100.)
