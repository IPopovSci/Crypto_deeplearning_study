import numpy as np


'Simple array for testing time series' \
'repeats sequence 1 2 3 4 5 4 3 2 1, n amount of times'
def dummy_timeseries(n):
    x = np.array([[1,2,3,4,5,4,3,2,1]])
    y = np.tile(x,n)
    print(y.shape)
    return y