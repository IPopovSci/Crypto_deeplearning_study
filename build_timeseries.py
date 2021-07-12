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

    print("length of time-series - inputs", x.shape)
    print("length of time-series - outputs", y.shape)

    return x, y