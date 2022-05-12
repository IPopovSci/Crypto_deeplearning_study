import numpy as np
np.random.seed(1337)
from Data_Processing.data_trim import trim_dataset
from tensorflow.keras.models import load_model
import os
from keras_self_attention import SeqSelfAttention
import tensorflow as tf
from pipeline.pipelineargs import PipelineArgs
from dotenv import load_dotenv
from Networks.network_config import NetworkParams
from Networks.losses_metrics import ohlcv_combined, metric_signs_close, ohlcv_cosine_similarity, ohlcv_mse, \
    assymetric_loss, assymetric_combined, metric_loss, metric_profit_ratio, profit_ratio_mse, profit_ratio_cosine, \
    profit_ratio_assymetric,assymetric_loss_mse
from Networks.custom_activation import cyclemoid
from Networks.custom_activation import p_swish,p_softsign
from keras.layers import Activation
import glob
from utility import remove_mean
from scipy.stats import spearmanr
from memory_profiler import profile
import gc

load_dotenv()

'''Module for training new models'''
pipeline_args = PipelineArgs.get_instance()
network_args = NetworkParams.get_instance()

batch_size = pipeline_args.args['batch_size']
time_steps = pipeline_args.args['time_steps']

'''Function to predict using existing model
Accepts: string model filename.
Returns: numpy array with predictions.'''


def predict(x_test_t, model_name='Default'):
    filepath = os.getenv('model_path') + f'/{pipeline_args.args["interval"]}/{pipeline_args.args["ticker"]}/{network_args.network["model_type"]}/' + model_name
    saved_model = load_model(filepath=filepath,
                             custom_objects={'metric_loss': metric_loss, 'assymetric_combined': assymetric_combined,
                                             'assymetric_loss': assymetric_loss,
                                             'metric_signs_close': metric_signs_close,
                                             'SeqSelfAttention': SeqSelfAttention, 'ohlcv_combined': ohlcv_combined,
                                             'ohlcv_cosine_similarity': ohlcv_cosine_similarity,
                                             'ohlcv_mse': ohlcv_mse,
                                             'metric_profit_ratio': metric_profit_ratio,
                                             'profit_ratio_mse': profit_ratio_mse,
                                             'profit_ratio_cosine': profit_ratio_cosine,
                                             'profit_ratio_assymetric': profit_ratio_assymetric,
                                             'cyclemoid': cyclemoid,
                                             'Activation': Activation,
                                             'p_swish': p_swish,
                                             'p_softsign':p_softsign,
                                             'assymetric_loss_mse':assymetric_loss_mse})
    saved_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00000005),
                        loss=ohlcv_combined, metrics=metric_signs_close)

    y_pred = saved_model.predict(trim_dataset(x_test_t[:], pipeline_args.args['batch_size']), batch_size=pipeline_args.args['batch_size'])

    #tf.keras.utils.plot_model(saved_model,'dense.png')

    #config = saved_model.get_config()

    # print(config['layers']['config'])
    # print(saved_model.optimizer.get_config())

    return y_pred

#Fix the slash directions on this one
def predict_average_ensembly(x_test_t,y_test_t):
    gc.collect()

    filepath = os.getenv('model_path') + f'/{pipeline_args.args["interval"]}/{pipeline_args.args["ticker"]}/ensembly_average/'

    all_models = os.path.join(filepath, '*.h5')

    all_models_list = glob.glob(all_models)

    pred_store = np.zeros([len(trim_dataset(x_test_t, pipeline_args.args['batch_size'])), 5])

    mean_count = np.full([len(trim_dataset(x_test_t, pipeline_args.args['batch_size'])), 5],len(all_models_list)) #To find average accurately, need to track how many models we use



    for model in all_models_list:

        model_name = model.split('ensemble/')[0]  # Grab just the name of the csv for saving purposes

        saved_model = load_model(filepath=model_name,
                                 custom_objects={'metric_loss': metric_loss, 'assymetric_combined': assymetric_combined,
                                                 'assymetric_loss': assymetric_loss,
                                                 'metric_signs_close': metric_signs_close,
                                                 'SeqSelfAttention': SeqSelfAttention, 'ohlcv_combined': ohlcv_combined,
                                                 'ohlcv_cosine_similarity': ohlcv_cosine_similarity,
                                                 'ohlcv_mse': ohlcv_mse,
                                                 'metric_profit_ratio': metric_profit_ratio,
                                                 'profit_ratio_mse': profit_ratio_mse,
                                                 'profit_ratio_cosine': profit_ratio_cosine,
                                                 'profit_ratio_assymetric': profit_ratio_assymetric,
                                                 'cyclemoid': cyclemoid,
                                                 'Activation': Activation,
                                                 'p_swish': p_swish,
                                                 'p_softsign':p_softsign,
                                                 'assymetric_loss_mse':assymetric_loss_mse})
        # saved_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00000005),
        #                     loss=ohlcv_combined, metrics=metric_signs_close)


        #print(f'predicting on model {model_name}') #Debug to see what model is running
        y_pred = saved_model.predict(trim_dataset(x_test_t, pipeline_args.args['batch_size']),use_multiprocessing=True,batch_size=pipeline_args.args['batch_size'])

        try: #This is to load in models with 3 output dimensions. Not ideal, will need a module to deal with it later
            y_pred = y_pred[:, -1, :]
        except:
            ''

        #Weighting by correlation, and inversion of negative
        for i in range(0,5):

            y_pred_coll = y_pred[:,i]

            coef_r, p_r = spearmanr(trim_dataset(y_test_t[:, i],pipeline_args.args['batch_size']), y_pred_coll)



            if coef_r < 0: #This will inverse models that are inversely-correlated
                y_pred_coll = -1 * y_pred_coll

                y_pred[:,i] = y_pred_coll

            if p_r > 0.05: #This will drop samples that are not well correlated
                y_pred_coll = 0
                y_pred[:, i] = y_pred_coll

                mean_count[:,i] -= 1 #since we won't be using this lag in this model, get rid it from average calc
        #print(mean_count)

        pred_store = pred_store + y_pred

    tf.keras.backend.clear_session()

    result = pred_store / mean_count


    return result

'''This function is supposed to be used as backtest for prediction
on old data through a loop;
Looping through .predict method causes memory leak
Left here just in case, do not use.'''
def predict_test(x_test_t,y_test_t):
    gc.collect()
    largest_lag = 49 + 39 + 36 + 37 + 26 + 37 + 44 + 35 + 38 + 38 + 41 + 43
    i = -1
    j = -129

    pred_store = np.zeros([1,5])

    x_test_t = tf.convert_to_tensor(x_test_t)

    while j-largest_lag > j-largest_lag-50:
        tf.keras.backend.clear_session()

        y_pred = predict_average_ensembly(x_test_t[j-largest_lag:i-largest_lag], y_test_t[j-largest_lag:i-largest_lag]) #last point 49 hours ago
        y_pred_mean = y_pred - np.mean(y_pred, axis=0)
        y_true_sign = np.sign(trim_dataset(y_test_t[j-largest_lag:i-largest_lag],pipeline_args.args['batch_size'])) #last point 1 hour ago
        y_pred_sign = np.sign(y_pred_mean)

        y_total_sign = y_true_sign[-1,:] * y_pred_sign[-1,:]
        #print(y_total_sign)

        pred_store = pred_store + y_total_sign
        print(pred_store)
        j -= 1
        i -= 1
        gc.collect()
        #print(gc.get_objects())