from keras import backend as K
from keras.layers import Layer
import tensorflow as tf


'''    y_true[0] = 1 hour
    y_true[1] = 4 hours
    y_true[2] = 12 hours
    y_true[3] = 24 hours
    y_true[4] = 48 hours'''

y_true = [2,0,-1,4,5,-3,2]
y_pred = [-1,2,3,0,-1,-3,1]

def vectorized_backtesting_loss(y_true,y_pred):
    # Generate buy and sell signals
    signal = tf.where(tf.math.greater(y_pred,0),1,-1)
    profit = y_pred

    #Calculate profit/loss made if buy/sell at predicted values
    #Calculate cumulative returns
    #loss = cum_strategy_returns - real_cum_returns
    loss = 1
    return signal

print(vectorized_backtesting_loss(y_true,y_pred))


class trade_on_close_loss_layer(Layer):
    def __init__(self, **kwargs):
        super(trade_on_close_loss_layer, self).__init__(**kwargs)
        self.loss_fn = 1

    # def build(self, input_shape):
    #     self.kernel = self.add_weight(name='kernel',
    #                                   shape=(input_shape[1], self.output_dim),
    #                                   initializer='normal', trainable=True)
    #     super(trade_on_close_loss_layer, self).build(input_shape)

    def call(self, y_true,y_pred):
        loss = self.loss_fn(y_true,y_pred)
        self.add_loss(loss,name='Portfolio return')
        return y_pred

    #def compute_output_shape(self, input_shape): return (input_shape[0], self.output_dim)