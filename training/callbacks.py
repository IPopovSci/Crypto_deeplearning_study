import os

import joblib
import tensorflow as tf
import tensorflow.keras.backend as K
from dotenv import load_dotenv
from tensorflow import keras
# tf.config.experimental_run_functions_eagerly(True)
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

from Arguments import args

load_dotenv()


def tf_diff_axis_1(a):
    return a[1:] - a[:-1]



def loss_unit_test(y_true_un,y_pred_un):


    y_pred_sign = math_ops.sign(y_pred_un)
    y_true_sign = math_ops.sign(y_true_un)
    abs_sign = math_ops.abs(math_ops.subtract(y_pred_sign,y_true_sign)) # 0 if same, 2 if different


    loss = tf.keras.losses.mean_absolute_error(y_true_un, y_pred_un)
    fin_loss = math_ops.multiply(abs_sign,math_ops.abs(math_ops.subtract(y_true_un,y_pred_un))+100)*10 + math_ops.abs(math_ops.subtract(y_true_un,y_pred_un))
    #print(fin_loss)


batch_size = args['batch_size']
MM_path = os.getenv('MM_Path')
SS_path = os.getenv('SS_Path')

mm_y = joblib.load(MM_path + ".y")
sc_y = joblib.load(SS_path + ".y")

mm_x = joblib.load(MM_path + ".x")
sc_x = joblib.load(SS_path + ".x")
def metric_signs(y_true,y_pred):

    y_true = ops.convert_to_tensor_v2(y_true)
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    #

#----------------
    # y_true = y_true[:,0]
    # y_pred = y_pred[:,0]


    # y_true = y_true[:,:,:,-1]
    # y_pred = y_pred[:,:,:,-1]
    #
    # y_true = y_true[:,-1,:]
    # y_pred = y_pred[:,-1,:]
    print(y_pred.shape)
    print(y_true.shape)

#--------------

    y_true_un = (((y_true - K.constant(mm_y.min_)) / K.constant(mm_y.scale_))* K.constant(sc_y.scale_)) + K.constant(sc_y.mean_)


    y_pred_un = (((y_pred - K.constant(mm_y.min_)) / K.constant(mm_y.scale_)) * K.constant(sc_y.scale_)) + K.constant(sc_y.mean_)

    # y_true_un = y_true_un[:,3]
    # y_pred_un = y_pred_un[:,3]
    # print(y_pred_un.shape)
    # print(y_true_un.shape)


    # tf.print(y_true_un)
    # tf.print(y_pred_un)


    # y_true_diff = tf_diff_axis_1(y_true_un)
    #
    # y_pred_diff = tf_diff_axis_1(y_pred_un)
    #------------------------------

    y_true_sign = math_ops.sign(y_true_un)
    y_pred_sign = math_ops.sign(y_pred_un)

    metric = math_ops.divide(math_ops.abs(math_ops.subtract(y_true_sign,y_pred_sign)),2.)
    #---------------------

    # metric = K.switch(K.less_equal(y_true * y_pred, 0),
    #     y_true/y_true,
    #     0 * y_true
    #     )

    return math_ops.multiply(math_ops.divide(math_ops.subtract(float(batch_size),K.sum(metric)),float(batch_size)),100.)

def custom_cosine_similarity(y_true,y_pred):


    y_true = ops.convert_to_tensor_v2(y_true)
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    # print(y_pred)


    # y_true = y_true[:,-1,3,:]
    # y_pred = y_pred[:,-1,3,:]
    #
    # y_true_un = (((y_true - K.constant(mm_y.min_)) / K.constant(mm_y.scale_))* sc_y.scale_) + sc_y.mean_
    #
    #
    # y_pred_un = (((y_pred - K.constant(mm_y.min_)) / K.constant(mm_y.scale_)) * sc_y.scale_) + sc_y.mean_


    # y_true_diff = tf_diff_axis_1(y_true_un)
    #
    # y_pred_diff = tf_diff_axis_1(y_pred_un)

    # y_true = y_true[:, -1, :, -1]
    # y_pred = y_pred[:, -1, :, -1]
    # print(y_pred.shape)

    # --------------

    y_true_un = (((y_true - K.constant(mm_y.min_)) / K.constant(mm_y.scale_)) * K.constant(sc_y.scale_)) + K.constant(
        sc_y.mean_)

    y_pred_un = (((y_pred - K.constant(mm_y.min_)) / K.constant(mm_y.scale_)) * K.constant(sc_y.scale_)) + K.constant(
        sc_y.mean_)
    #
    # y_true_un = y_true_un[:, 3]
    # y_pred_un = y_pred_un[:, 3]

    #
    # tf.print(y_true_un.shape)
    # tf.print(y_pred_un.shape)





    loss = tf.keras.losses.cosine_similarity(y_true_un, y_pred_un, axis=-1)


    #metric = math_ops.divide(metric_signs(y_true,y_pred),100)


    return loss

def custom_mean_absolute_error(y_true,y_pred):


    y_true = ops.convert_to_tensor_v2(y_true)
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    metric = math_ops.divide(metric_signs(y_true, y_pred), 100)


    #
    y_true_un = (((y_true - K.constant(mm_y.min_)) / K.constant(mm_y.scale_))* sc_y.scale_) + sc_y.mean_


    y_pred_un = (((y_pred - K.constant(mm_y.min_)) / K.constant(mm_y.scale_)) * sc_y.scale_) + sc_y.mean_

    # y_true_un = y_true
    # y_pred_un = y_pred[:,-1]
    # y_pred_un = tf.reshape(y_pred_un, [-1, 1])



    # print(y_true_un[-1])
    # #
    # print(y_pred_un[:,-1])




    loss = tf.keras.losses.mean_absolute_error(y_true_un, y_pred_un)


    return K.mean(loss)

class ResetStatesOnEpochEnd(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.reset_states()
        print((self.model.output))
        print('states are reset!')