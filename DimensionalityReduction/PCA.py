import numpy as np
from time import time
from sklearn import decomposition

def fit_transform(X_highdimensional):
    '''
        Performs PCA on the given highdimensional dataset. The data needs to be a numpy array of the form (n_samples, n_features).

        X_highdimensional:  (n_samples, n_features) matrix containing the highdimensional datapoints.

        Returns a tuple (X_low, time) with the lowdimensional representation and the time the execution took.
    '''    
    pca = decomposition.PCA(n_components=2)
    t0 = time()
    X_low = pca.fit_transform(X_highdimensional)
    return X_low, (time() - t0)