import numpy as np
from time import time
from sklearn import decomposition

def fit_transform(X_highdimensional, t_kernel="linear"):
    '''
        Performs Kernel PCA on the given highdimensional dataset. The data needs to be a numpy array of the form (n_samples, n_features).

        X_highdimensional:  (n_samples, n_features) matrix containing the highdimensional datapoints.
        t_kernel: The type of kernel to be used. May be "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
                  For more details, see http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html
                  Default: linear, so the result is similar to PCA.

        Returns a tuple (X_low, time) with the lowdimensional representation and the time the execution took.
    '''
    kernel_pca = decomposition.KernelPCA(n_components=2, kernel=t_kernel)
    t0 = time()
    X_low = kernel_pca.fit_transform(X_highdimensional)
    return X_low, (time() - t0)