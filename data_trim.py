def trim_dataset(df, batch_size):
    # trims dataset to a size that's divisible by BATCH_SIZE
    no_of_rows_drop = df.shape[0] % batch_size

    if no_of_rows_drop > 0:
        return df[no_of_rows_drop:]
    else:
        return df

def trim_dataset_conv(df, batch_size):
    # trims dataset to a size that's divisible by BATCH_SIZE
    no_of_rows_drop = df.shape[1] % batch_size

    if no_of_rows_drop > 0:
        return df[:,no_of_rows_drop:,:,:]
    else:
        return df

