import os

import joblib
import tensorflow as tf
import tensorflow.keras.backend as K
from dotenv import load_dotenv
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from pipeline.pipelineargs import PipelineArgs
from utility import unscale


load_dotenv()

pipeline_args = PipelineArgs.get_instance()

batch_size = pipeline_args.args['batch_size']


'''Custom cosine similarity loss
Extracts the last time step and unscales y values before computing loss
Final result has 1. added to it so that the loss is normalized to 0-2 range'''
def ohlcv_cosine_similarity(y_true,y_pred):



    y_true = ops.convert_to_tensor_v2(y_true)
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)


    y_pred = y_pred[:,-1,:] #Because Dense predictions will have timesteps


    y_true_un,y_pred_un = unscale(y_true,y_pred)

    loss = tf.keras.losses.cosine_similarity(y_true_un, y_pred_un, axis=-1)



    return loss + 1.0

'''Custom MSE loss
Extracts the last time step and unscales y values before computing loss'''
def ohlcv_mse(y_true,y_pred):
    y_true = ops.convert_to_tensor_v2(y_true)
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)


    y_pred = y_pred[:,-1,:] #Because Dense predictions will have timesteps


    y_true_un,y_pred_un = unscale(y_true,y_pred)


    #print(y_true_un.shape,y_pred_un.shape)

    loss = K.mean(K.square(y_true_un - y_pred_un), axis=-1)

    return loss

'''Combined cosine similarity and MSE loss
CS loss is squared to put a higher emphasis on the correct direction'''
def ohlcv_combined(y_true,y_pred):
    loss = (ohlcv_mse(y_true,y_pred) * (ohlcv_cosine_similarity(y_true,y_pred) ** 3))

    return loss

def metric_signs_close(y_true,y_pred):

    y_true = ops.convert_to_tensor_v2(y_true)
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)


    y_pred = y_pred[:, -1, :]  # Because Dense predictions will have timesteps




    y_true_un, y_pred_un = unscale(y_true, y_pred)

    y_true_un = tf.expand_dims(y_true_un[:, 3], axis=1)  # This metric works on closing
    y_pred_un = tf.expand_dims(y_pred_un[:, 3], axis=1)

    #print(y_pred_un.shape, y_true_un.shape)



    y_true_sign = math_ops.sign(y_true_un)
    y_pred_sign = math_ops.sign(y_pred_un)

    metric = math_ops.divide(math_ops.abs(math_ops.subtract(y_true_sign,y_pred_sign)),2.)

    return math_ops.multiply(math_ops.divide(math_ops.subtract(float(batch_size),K.sum(metric)),float(batch_size)),100.)