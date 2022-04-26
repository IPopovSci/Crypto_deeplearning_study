import numpy as np
import matplotlib.pyplot as plt
import os
from Networks.network_config import NetworkParams
import numpy as np
from pipeline.pipelineargs import PipelineArgs
from dotenv import load_dotenv
from utility import remove_mean
import pandas as pd
import tensorflow.keras.backend as K
import tensorflow as tf

def graph_loss():
    y_true_pos = np.full(1000,0.5)
    y_true_neg = np.full(1000, -0.5)

    y_pred = np.arange(-5,5,10/1000,dtype='float32')

    print(y_pred)


    def assymetric_loss_mse(y_true,y_pred):

        alpha = 100.
        loss = K.switch(K.less(y_true * y_pred, 0),
                        alpha * y_pred**2 + K.square(y_true-y_pred),
                        K.square(y_true-y_pred)
                        )
        return np.array(loss)

    def metric_profit_ratio(y_true, y_pred):

        ratio = tf.math.divide_no_nan(y_pred, y_true)
        loss = K.switch(K.greater_equal(K.abs(ratio), 1.), tf.math.divide_no_nan(1., ratio), ratio)
        return np.array(-loss + 1)

    def ohlcv_cosine_similarity(y_true, y_pred):

        loss = -np.sum(np.linalg.norm(y_true,2) * np.linalg.norm(y_pred,2))

        return np.array(loss + 1.0)


    loss_pos = assymetric_loss_mse(y_true_pos,y_pred) + metric_profit_ratio(y_true_pos,y_pred) + ohlcv_cosine_similarity(y_true_pos,y_pred)
    loss_neg = assymetric_loss_mse(y_true_neg,y_pred) + metric_profit_ratio(y_true_neg,y_pred) + ohlcv_cosine_similarity(y_true_neg,y_pred)

    plt.plot(y_pred,loss_pos, label='Loss With Positive True Value')
    plt.plot(y_pred,loss_neg, label='Loss With Negative True Value')
    plt.title('Sum of 3 losses')

    plt.legend()
    plt.show()

graph_loss()