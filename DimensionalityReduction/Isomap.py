import numpy as np
from time import time
from sklearn import manifold

def fit_transform(X_highdimensional, n_neighbors=5):
    '''
        Performs Isomap on the given highdimensional dataset. The data needs to be a numpy array of the form (n_samples, n_features).

        X_highdimensional:  (n_samples, n_features) matrix containing the highdimensional datapoints.
        n_neighbors: The number of neighbours to consider for each point. Default: 5

        Returns a tuple (X_low, time, error) with the lowdimensional representation and the time the execution took, and the reconstruction error.
    '''
    isomap = manifold.Isomap(n_neighbors, n_components=2, eigen_solver="dense")
    t0 = time()
    X_low =  isomap.fit_transform(X_highdimensional);
    return X_low, (time() - t0), isomap.reconstruction_error()