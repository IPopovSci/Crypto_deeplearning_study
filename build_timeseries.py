import pandas as pd
import numpy as np
from Arguments import args

'''This takes training or test data and returns x,y scaled using window approach (Unless TIME_STEPS = 1)'''
def build_timeseries(mat, y_col_index):
    TIME_STEPS = args["time_steps"]


    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]

    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))

    print("Length of inputs", dim_0)

    for i in range(dim_0):
        x[i] = mat[i:TIME_STEPS + i]
        y[i] = mat[TIME_STEPS + i, y_col_index]
    '''Step 10 - offset target values'''
    from data_shift import shift
    y = shift(y,1,fill_value=0)

    print("length of time-series - inputs", x.shape)
    print("length of time-series - outputs", y.shape)

    return x, y

def build_timeseries_conv(mat):
    TIME_STEPS = args["time_steps"]


    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]

    x = np.zeros((dim_0, TIME_STEPS, dim_1))

    # print(x.shape)

    print("Length of inputs", dim_0)

    for i in range(dim_0):
        x[i] = mat[i:TIME_STEPS + i]
    print("length of time-series - inputs", x.shape)
    x = np.transpose(x,(1,0,2))
    x = np.expand_dims(x, axis=-1)
    print("length of time-series - inputs", x.shape)

    return x