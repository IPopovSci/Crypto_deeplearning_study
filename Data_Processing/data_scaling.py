from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from pipeline.pipelineargs import PipelineArgs

'''Standard scaler transform - it will substract the mean to center the data
as well as bring the standard deviation to 1. Will transform incoming pandas data into numpy in the process
The target columns are the first 5
returns transformed train,validation,test sets as well as the scaler
If mode = training, will create new scalers, otherwise will read them from the drive'''

pipeline_args = PipelineArgs.get_instance()

def SS_transform(x_train, x_validation, x_test, y_train, y_validation, y_test, mode='training',interval='1h',ticker='ethusdt', SS_path=[]):
    if mode == 'prediction':
        sc_x = joblib.load(SS_path + f'\{interval}' + f'\\{ticker}' + "\ss.x")
        sc_y = joblib.load(SS_path + f'\{interval}' + f'\\{ticker}' + "\ss.y")

        x_train_ss = sc_x.transform(x_train)

        x_validation_ss = sc_x.transform(x_validation)

        x_test_ss = sc_x.transform(x_test)

        y_train_ss = sc_y.transform(y_train)

        y_validation_ss = sc_y.transform(y_validation)

        y_test_ss = sc_y.transform(y_test)

    elif mode == 'training':
        sc_x = StandardScaler()
        sc_y = StandardScaler()

        x_train_ss = sc_x.fit_transform(x_train)

        x_validation_ss = sc_x.transform(x_validation)

        x_test_ss = sc_x.transform(x_test)

        y_train_ss = sc_y.fit_transform(y_train)

        y_validation_ss = sc_y.transform(y_validation)

        y_test_ss = sc_y.transform(y_test)

        if not os.path.exists(SS_path + f'\{interval}' + f'\\{ticker}'):
            os.makedirs(SS_path + f'\{interval}' + f'\\{ticker}',mode= 0o777)


        joblib.dump(sc_x, SS_path + f'\{interval}' + f'\\{ticker}' + "\ss.x")
        joblib.dump(sc_y, SS_path + f'\{interval}' + f'\\{ticker}' + "\ss.y")

    pipeline_args.args['ss_x_path'] = SS_path + f'\{interval}' + f'\\{ticker}' + "\ss.x"
    pipeline_args.args['ss_y_path'] = SS_path + f'\{interval}' + f'\\{ticker}' + "\ss.y"


    return x_train_ss, x_validation_ss, x_test_ss, y_train_ss, y_validation_ss, y_test_ss, sc_y


'''For transfer learning on small datasets, no validation, no test'''


def min_max_transform(x_train, x_validation, x_test, y_train, y_validation, y_test, mode='training',interval='1h',ticker='ethusdt',
                      MM_path=[]):  # old version doesn't do robust scaling, use when predding from older models
    if mode == 'prediction':
        mm_x = joblib.load(MM_path + f'\{interval}' + f'\\{ticker}' + "\mm.x")
        mm_y = joblib.load(MM_path + f'\{interval}' + f'\\{ticker}' + "\mm.y")

        x_train = mm_x.transform(x_train)

        x_validation = mm_x.transform(x_validation)

        x_test = mm_x.transform(x_test)

        y_train = mm_y.transform(y_train)

        y_validation = mm_y.transform(y_validation)

        y_test = mm_y.transform(y_test)


    elif mode == 'training':
        mm_x = MinMaxScaler(feature_range=(-1, 1))

        mm_y = MinMaxScaler(feature_range=(-1, 1))

        x_train = mm_x.fit_transform(x_train)

        x_validation = mm_x.transform(x_validation)

        x_test = mm_x.transform(x_test)

        y_train = mm_y.fit_transform(y_train)

        y_validation = mm_y.transform(y_validation)

        y_test = mm_y.transform(y_test)

        if not os.path.exists(MM_path + f'\{interval}' + f'\\{ticker}'):
            os.makedirs(MM_path + f'\{interval}' + f'\\{ticker}\\',mode= 0o777)

        joblib.dump(mm_x, MM_path + f'\{interval}' + f'\\{ticker}' + "\mm.x")
        joblib.dump(mm_y, MM_path + f'\{interval}' + f'\\{ticker}' + "\mm.y")

    pipeline_args.args['mm_x_path'] = MM_path + f'\{interval}' + f'\\{ticker}' + "\mm.x"
    pipeline_args.args['mm_y_path'] = MM_path + f'\{interval}' + f'\\{ticker}' + "\mm.y"

    return x_train, x_validation, x_test, y_train, y_validation, y_test, mm_y