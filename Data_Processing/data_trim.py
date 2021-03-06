'''This trims the dataset so that its row size is divisible by the batch size, needed for stateful Networks.
Accepts: dataframe, int batch size
Returns: dataframe'''


def trim_dataset(df, batch_size):
    # trims dataset to a size that's divisible by BATCH_SIZE
    no_of_rows_drop = df.shape[0] % batch_size

    if no_of_rows_drop > 0:
        return df[no_of_rows_drop:]
    else:
        return df
