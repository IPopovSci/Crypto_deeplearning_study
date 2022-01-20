import numpy as np
from Arguments import args
from Data_Processing.data_trim import trim_dataset

'''This takes training or test data and returns x,y scaled using window approach (Unless TIME_STEPS = 1)'''
def build_timeseries(x_t,y_t):
    #Shifting the y data 1 day in advance
    x_t = x_t[:-1]
    y_t = y_t[1:]

    TIME_STEPS = args["time_steps"]
    # print(x_t.shape)
    # print(y_t.shape) #So we have 5 outputs now, need a new loop

    dim_0 = x_t.shape[0] - TIME_STEPS
    dim_1 = x_t.shape[1]
    print(dim_0,TIME_STEPS,dim_1)


    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,5))

    print("Length of inputs", dim_0)


    for i in range(dim_0):
        x[i] = x_t[i:TIME_STEPS + i]
        #y[i] = y_t[TIME_STEPS + i]
    for column in range(5):
        for row in range(dim_0):
            y[row,column] = y_t[TIME_STEPS + row,column]

    x = trim_dataset(x,batch_size=args['batch_size'])
    y = trim_dataset(y, batch_size=args['batch_size'])




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