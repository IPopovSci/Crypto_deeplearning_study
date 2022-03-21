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

def correct_signs(y_true,y_pred):
    y_total = np.empty(5)
    if pipeline_args.args['expand_dims'] == False:
        y_pred = y_pred[:,-1,:]


    for i in range(5):
        y_true_sign = np.sign(y_true[:,i])
        y_pred_sign = np.sign(y_pred[:,i])

        y_total_sign = np.multiply(y_true_sign,y_pred_sign)

        y_total[i] = np.sum(y_total_sign)

        print(f'{pipeline_args.args["data_lag"][-i-1]}h correct amount of signs is: {y_total[i]}')


def information_coefficient(y_true,y_pred):
    coef_r, p_r = spearmanr(y_true, y_pred)
    alpha = 0.05
    if p_r < alpha:
        print('Samples are correlated (reject H0) p=%.3f' % p_r)
        print('Spearmans correlation coefficient: %.3f' % coef_r)
    else:
        print('Samples are un-correlated (Fail to reject H0) p=%.3f' % p_r)
        print('Spearmans correlation coefficient: %.3f' % coef_r)

    return coef_r,p_r

def plot_backtest(y_test_t,y_pred):
    if pipeline_args.args['expand_dims'] == False:
        y_pred = y_pred[:,-1,:] #Because Dense predictions will have timesteps



def ic_coef(y_true,y_pred):
    if pipeline_args.args['expand_dims'] == False:
        y_pred = y_pred[:,-1,:]

    for i in range(5):
        print(f'{pipeline_args.args["data_lag"][-i-1]}h lag spearman statistics:')
        information_coefficient(y_true[:,i],y_pred[:,i])




