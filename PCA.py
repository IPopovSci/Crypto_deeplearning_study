import pandas as pd
from sklearn.decomposition import PCA
from Arguments import args
import numpy as np

'''PCA will de-correlate variables allowing downstreaming algorithms to predict more accurately'''


def pca_reduction(x_train, x_validation, x_test):
    pca = PCA(n_components='mle',svd_solver='full',whiten=True)  #Auto-solve for number of components



    pca_reduce_train = pca.fit_transform(x_train)
    pca_reduce_test = pca.transform(x_test)
