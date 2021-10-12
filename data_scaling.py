from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler,RobustScaler
import pandas as pd
from Arguments import args
import joblib
import numpy as np
import os

'''Standard scaler transform - it will substract the mean to center the data
as well as bring the standard deviation to 1. Will transform incoming pandas data into numpy in the process
The target columns are the first 5
returns transformed train,validation,test sets as well as the scaler'''
def SS_transform(x_train,x_validation,x_test):

    sc = StandardScaler()

    x_train_ss = sc.fit_transform(x_train)

    x_validation_ss = sc.transform(x_validation)

    x_test_ss = sc.transform(x_test)

    return x_train_ss,x_validation_ss,x_test_ss,sc

'''Min-max Scaler for convering values to (-1,1) domain
y values are converted based on y training data'''

def min_max_transform(x_train,x_validation,x_test,y_train, y_validation, y_test): #old version doesn't do robust scaling, use when predding from older models

    mm = MinMaxScaler(feature_range=(-1,1)) #Define scaler

    x_train = mm.fit_transform(x_train)

    x_validation = mm.transform(x_validation)

    x_test = mm.transform(x_test)

    y_train = mm.fit_transform(y_train)

    y_validation = mm.transform(y_validation)

    y_test = mm.transform(y_test)

    return x_train,x_validation,x_test,y_train, y_validation, y_test, mm
