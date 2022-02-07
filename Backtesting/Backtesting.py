import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

def correct_signs(y_true,y_pred):
    try:
        y_true_sign = np.sign(y_true[:,3])
        y_pred_sign = np.sign(y_pred[:,3])
    except:
        y_true_sign = np.sign(y_true[:])
        y_pred_sign = np.sign(y_pred[:])

    y_total_sign = np.multiply(y_true_sign,y_pred_sign)

    y_total = np.sum(y_total_sign)

    return y_total