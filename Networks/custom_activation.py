import math as m
import tensorflow as tf
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects

def cyclemoid(x):
    """
    Implementation tweaked from PyTorch
    implementation, to be used as functions & in Lambda layers
    """
    pi = tf.constant(m.pi)
    term1 = tf.math.tanh(pi * x)
    term2 = tf.math.tanh(pi * tf.square(x) - 0.95) ** 2
    term3 = (
        tf.math.cos(tf.clip_by_value(x, clip_value_min=-3.0, clip_value_max=3.0)) ** 2
    )
    return term1 * term2 * term3

get_custom_objects().update({'cyclemoid': Activation(cyclemoid)})