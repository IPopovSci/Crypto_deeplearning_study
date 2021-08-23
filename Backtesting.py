import math
import numpy as np


def up_or_down(y_preds):
    if y_preds[-1] > y_preds[-2]:
        percent = (y_preds[-1] - y_preds[-2]) / y_preds[-1] * 100
        print('Up by', percent)
    if y_preds[-1] < y_preds[-2]:
        percent = (y_preds[-1] - y_preds[-2]) / y_preds[-1] * 100
        print('Down by', percent)


def back_test(y_preds, y_true):
    i = -1
    prediction_list = []
    true_list = []
    same_sign_list = []
    while i > -len(y_preds) + 1:
        prediction = (y_preds[i] - y_preds[i - 1]) / y_preds[i] * 100
        true = (y_true[i] - y_true[i - 1]) / y_true[i] * 100
        if abs(prediction) + abs(true) == abs(prediction + true):
            same_sign_list.append('1')
        else:
            same_sign_list.append('0')
        prediction_list.append(prediction)
        true_list.append(true)

        i -= 1
    return print('Predicted correct:', same_sign_list.count('1'), 'Predicted wrong:', same_sign_list.count('0'),
                 'Percent_total_true:', sum(true_list), 'Percent_total_Preds:', sum(prediction_list),
                 'Prediction for today:', (y_preds[-2] - y_preds[-3]) / y_preds[-3] * 100, 'Prediction for tomorrow:',
                 (y_preds[-1] - y_preds[-2]) / y_preds[-2] * 100)
