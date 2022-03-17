import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from scipy.stats import spearmanr,kendalltau

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


def information_coefficient(y_true,y_pred):
    coef_r, p = spearmanr(y_true, y_pred)
    print('Spearmans correlation coefficient: %.3f' % coef_r)
    alpha = 0.05
    if p > alpha:
        print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    else:
        print('Samples are correlated (reject H0) p=%.3f' % p)

    coef, p = kendalltau(y_true, y_pred)
    print('Kendall correlation coefficient: %.3f' % coef)
    # interpret the significance
    alpha = 0.05
    if p > alpha:
        print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    else:
        print('Samples are correlated (reject H0) p=%.3f' % p)




