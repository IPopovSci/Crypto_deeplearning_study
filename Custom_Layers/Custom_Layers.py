from keras import backend as K
from keras.layers import Layer
from training.callbacks import portfolio_metric
import tensorflow as tf

'''    y_true[0] = Open
    y_true[1] = High
    y_true[2] = Low
    y_true[3] = Close
    y_true[4] = Volume'''

class trade_on_close_loss_layer(Layer):
    def __init__(self, **kwargs):
        super(trade_on_close_loss_layer, self).__init__(**kwargs)
        self.metric_fn = portfolio_metric()
        K.binary_crossentropy

    # def build(self, input_shape):
    #     self.kernel = self.add_weight(name='kernel',
    #                                   shape=(input_shape[1], self.output_dim),
    #                                   initializer='normal', trainable=True)
    #     super(trade_on_close_loss_layer, self).build(input_shape)

    def call(self, y_true,y_pred):
        metric = self.metric_fn(y_true,y_pred)
        self.add_metric(metric,name='Portfolio on close')
        return y_pred

    #def compute_output_shape(self, input_shape): return (input_shape[0], self.output_dim)