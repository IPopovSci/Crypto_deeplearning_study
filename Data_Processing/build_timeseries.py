import numpy as np
from Data_Processing.data_trim import trim_dataset
from numpy import expand_dims
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from pipeline.pipelineargs import PipelineArgs
from dotenv import load_dotenv
import os

pipeline_args = PipelineArgs.get_instance()
load_dotenv()

'''This function takes training or test data and returns x,y scaled using window approach (Unless TIME_STEPS = 1)
if expand_dims = True will add a singular channel dimension to the end of the data.
Accepts: x and y 2 dimensional (samples,outcomes) dataset; TIME_STEPS, batch_size and expand_dims str parameters.
Returns: x and y dataset 3 dimensional (samples, TIME_STEPS, outcomes) dataset.'''


def build_timeseries(x_t, y_t, TIME_STEPS, batch_size, expand_dims=False):
    print('before timeseries conversion', x_t.shape)

    dim_0 = x_t.shape[0] - TIME_STEPS
    print('dim 0 is ',dim_0)
    dim_1 = x_t.shape[1]
    pipeline_args.args['num_features'] = dim_1  # Final number of features, useful later for network creation

    x = np.zeros((dim_0, TIME_STEPS, dim_1))

    y = np.zeros((dim_0, 5))

    if os.environ['data_window'] == 'sliding':
        for i in range(dim_0):
            x[i] = x_t[i + 1:TIME_STEPS + i + 1]
            y[i] = y_t[TIME_STEPS + i]

            #debug
            # if i in (0, dim_0 - 1):
            #     print(f"x at {i} = [{i + 1}:{TIME_STEPS + i + 1}]")
            #     print(f"y at {i} = {TIME_STEPS + i}")

    # elif os.environ['data_window'] == 'stateful':
    #         #dim_0 = x_t.shape[0]
    #         for i in range(int(dim_0 / int(TIME_STEPS))):
    #             x[i] = x_t[i * TIME_STEPS:i * TIME_STEPS + TIME_STEPS]
    #             #print(x)
    #             y[i] = y_t[i * TIME_STEPS + TIME_STEPS]
    # print('x_t values', x_t[-5:, 15])
    # print('x values', x[-5:, -1, 15])
    # print(y)


    x = trim_dataset(x, batch_size=batch_size)
    y = trim_dataset(y, batch_size=batch_size)

    if expand_dims == True:
        x = np.expand_dims(x, axis=-1)

    print("length of time-series - inputs", x.shape)
    print("length of time-series - outputs", y.shape)

    return x, y
