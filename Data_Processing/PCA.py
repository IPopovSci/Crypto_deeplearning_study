from sklearn.decomposition import PCA

'''PCA will de-correlate variables allowing downstream algorithms to predict more accurately'''

def pca_reduction(x_train, x_validation, x_test):
    pca = PCA(n_components=82, svd_solver='auto', whiten=True)  # Auto-solve for number of components

    pca_train = pca.fit_transform(x_train)
    pca_test = pca.transform(x_test)
    pca_validation = pca.transform(x_validation)

    return pca_train, pca_validation, pca_test
