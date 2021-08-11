import numpy as np


def row_difference(df):
    df_diff = np.empty((df.shape[0], df.shape[1]))
    for column in range(df.shape[1]):
        # print('Now working on column:', column)
        for row in range(df.shape[0]):
            if row != 0:
                df_diff[row, column] = df[row, column] - df[row - 1, column]
    return df_diff
