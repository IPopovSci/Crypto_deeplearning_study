
from Data_Processing.get_data import ticker_data
import pandas as pd
from pipeline_args import args

import numpy as np
from Data_Processing.data_trim import trim_dataset_conv
from Data_Processing.ta_feature_add import add_ta
from Visual_Network import data_proc
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale,scale
from Data_Processing.build_timeseries import build_timeseries_conv
from Old_and_crap.Visual_Network import LSTM_network_conv

ticker = args['ticker']

'''Step 1 - Download stock price data from yahoo finance'''
ticker_data(ticker)
'''Step 2 - Add TA Analysis'''
add_ta(ticker)
'''Step 3'''
'''Loading the dataset and converting it into price differences'''
df = pd.read_csv(f'data/02_ta/{ticker}.csv')

df = np.array(df) #Convert into numpy array
df = np.delete(df, 0, 1)  # This will remove the date field
# df = np.delete(df,0,1) #This will remove VIX - Why is it empty? (Its only empty for the first n rows)

x_t = data_proc.row_difference(df) #This will populate an empty numpy array with differences in values;Last column is the price column - I double checked
"""Step 4"""
'''Scaling data so PCA has easier time working with it'''
''''''
x_t = scale(x_t)  # This, reportedly, scales one feature at a time
y_t = x_t[:,-1:] #This is only the last column I.e prices
x_t = x_t[:,:-1] #We don't want to PCA out y values

'''Step 5 - use PCA reduction to reduce number of features (Faster training, less feature burden on the model'''
np.nan_to_num(x_t,copy=False) #get rid of nans
pca = PCA(n_components=0.99,svd_solver = 'full') #This will create a PCA object to apply to our x data
x_t = pca.fit_transform(x_t) #Applying to x data



'''Step 6'''
'''Min_max everything to between 0 and 1, then multiply by 255 to get the pixel value'''
print(x_t.shape)
x_t = np.concatenate((x_t,y_t),axis=1) #Now that we skipped PCA for y values, we can return the array to having the price in it

x_t = minmax_scale(x_t, feature_range=(0, 1))  # This, reportedly, scales one feature at a time
x_t = (x_t * 255).astype(np.uint8) #Converting everything to grayscale values of 0-1 * 255
'''Step 7'''
'''Build Timeseries out of data'''
x_t= build_timeseries_conv(x_t) #This will create sliding windows, add one extra channel dimension, as well as transpose to bring it to the right dimension order - (Num_steps, width, height, channels)
print('x_t',x_t.shape)
x_t = x_t[:,-100:,:] #Grabbing only 500 samples for now
'''Step 8'''
"""Split the data into training/testing"""
indexes = np.arange(x_t.shape[1])
#np.random.shuffle(indexes)
train_index = indexes[: int(0.9 * x_t.shape[1])]
val_index = indexes[int(0.9 * x_t.shape[1]) :]
train_dataset = x_t[:,train_index,:,:]
val_dataset = x_t[:,val_index,:,:]

'''Step 9'''
''' We'll define a helper function to shift the frames, where
 `x` is frames 0 to n - 1, and `y` is frames 1 to n.'''
print('train_dataset shape:',train_dataset.shape,'val_dataset shape:',val_dataset.shape)
def create_shifted_frames(data):
    x = data[:, 0 : data.shape[1] - 1,: , :]
    y = data[:,  1 : data.shape[1],:, :]
    return x, y
batch_size = args['batch_size']
# Apply the processing function to the datasets.
x_train, y_train = create_shifted_frames(train_dataset)
x_val, y_val = create_shifted_frames(val_dataset)
x_train, y_train, x_val, y_val = trim_dataset_conv(x_train,batch_size),trim_dataset_conv(y_train,batch_size),trim_dataset_conv(x_val,batch_size),trim_dataset_conv(y_val,batch_size)
x_train = x_train[None,:,:,:,:]
y_train = y_train[None,:,:,:,:]
x_val = x_val[None,:,:,:,:]
y_val = y_val[None,:,:,:,:]
print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))
# '''Step 10 - data visualization
# Something is currently wrong here - either my data is not right, or the way I'm plotting it is not right'''
# # Construct a figure on which we will visualize the images.
# fig, axes = plt.subplots(5, 6, figsize=(100, 80))
#
# # Plot each of the sequential images for one random data example.
# data_choice = np.random.choice(range(len(train_dataset[2])), size=1)[0]
# for idx, ax in enumerate(axes.flat):
#     ax.imshow(train_dataset[idx][data_choice], cmap="gray")
#     ax.set_title(f"Frame {idx + 1}")
#     ax.axis("off")
#
# # Print information and display the figure.
# #print(f"Displaying frames for example {data_choice}.")
# plt.show()

'''Step 11 - Create model'''

model = LSTM_network_conv.LSTM_network_conv_create(x_train)

'''Step 12 - Model training'''

# Define some callbacks to improve training.
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5) #Wow cool function!

# Define modifiable training hyperparameters.
epochs = 20
batch_size = args['batch_size']

# Fit the model to the training data.
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, reduce_lr],
)