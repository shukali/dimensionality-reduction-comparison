import numpy as np
from time import time
from sklearn import manifold

def fit_transform(X_highdimensional, n_neighbors=5):
    '''
        Performs LTSA on the given highdimensional dataset. The data needs to be a numpy array of the form (n_samples, n_features).

        X_highdimensional:  (n_samples, n_features) matrix containing the highdimensional datapoints.
        n_neighbors: The number of neighbours to consider for each point. Default: 5

        Returns a tuple (X_low, time, err) with the lowdimensional representation, the time the execution took and the error.
    '''
    ltsa = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='ltsa', eigen_solver="dense")
    t0 = time()
    X_low = ltsa.fit_transform(X_highdimensional)
    return X_low, (time() - t0), ltsa.reconstruction_error_