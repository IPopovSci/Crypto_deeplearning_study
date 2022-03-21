import numpy as np
from Data_Processing.data_trim import trim_dataset
from numpy import expand_dims
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from pipeline.pipelineargs import PipelineArgs



'''This takes training or test data and returns x,y scaled using window approach (Unless TIME_STEPS = 1)'''
pipeline_args = PipelineArgs.get_instance()
'''Currently this does not work for predictions - we don't get the last data_lag amount of x data - '''
def build_timeseries(x_t,y_t,TIME_STEPS,batch_size,expand_dims=False,data_lag=1):

    print('before timeseries conversion',x_t.shape)

    dim_0 = x_t.shape[0] - TIME_STEPS
    dim_1 = x_t.shape[1]
    pipeline_args.args['num_features'] = dim_1 #Final number of features, useful later for network creation

    x = np.zeros((dim_0, TIME_STEPS, dim_1))

    y = np.zeros((dim_0, 5))


    for i in range(dim_0):
        x[i] = x_t[i:TIME_STEPS + i]

        y[i] = y_t[TIME_STEPS + i]




    x = trim_dataset(x,batch_size=batch_size)
    y = trim_dataset(y, batch_size=batch_size)

    if expand_dims == True:
        x = np.expand_dims(x,axis = -1)

    print("length of time-series - inputs", x.shape)
    print("length of time-series - outputs", y.shape)


    return x, y

'++-----++++-+--+---'