import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from scipy.stats import spearmanr,kendalltau
from pipeline.pipelineargs import PipelineArgs
from dotenv import load_dotenv
from Networks.network_config import NetworkParams
import pandas as pd

import os
load_dotenv()
pipeline_args = PipelineArgs.get_instance()

'''Function that calculates the amount of equal signs between y_true and y_pred
Accepts: y_true,y_pred.
Returns: Print out of the information.'''

def correct_signs(y_true,y_pred):
    # if pipeline_args.args['expand_dims'] == False:
    #     y_pred = y_pred[:,-1,:]
    y_total = np.empty(5)
    y_total_mean = np.empty(5)
    y_pred_mean = y_pred - np.mean(y_pred,axis=0)



    for i in range(5):
        y_true_sign = np.sign(y_true[:,i])
        y_pred_sign = np.sign(y_pred[:,i])

        y_total_sign = np.multiply(y_true_sign,y_pred_sign)

        y_total[i] = np.sum(y_total_sign)


        y_pred_sign_mean = np.sign(y_pred_mean[:,i])

        y_total_sign_mean = np.multiply(y_true_sign,y_pred_sign_mean)

        y_total_mean[i] = np.sum(y_total_sign_mean)




        print(f'{pipeline_args.args["data_lag"][-i-1]}h correct amount of signs is: {y_total[i]}')
        print(f'{pipeline_args.args["data_lag"][-i - 1]}h correct amount of signs with mean removal is: {y_total_mean[i]}')


def information_coefficient(y_true,y_pred,verbose=True):
    coef_r, p_r = spearmanr(y_true, y_pred)
    alpha = 0.05

    if verbose:
        if p_r < alpha:
            print('Samples are correlated (reject H0) p=%.3f' % p_r)
            print('Spearmans correlation coefficient: %.3f' % coef_r)
        else:
            print('Samples are un-correlated (Fail to reject H0) p=%.3f' % p_r)
            print('Spearmans correlation coefficient: %.3f' % coef_r)

    return coef_r,p_r

def ic_coef(y_true,y_pred):
    # if pipeline_args.args['expand_dims'] == False:
    #     y_pred = y_pred[:,-1,:]

    for i in range(5):
        print(f'{pipeline_args.args["data_lag"][-i-1]}h lag spearman statistics:')
        information_coefficient(y_true[:,i],y_pred[:,i])




