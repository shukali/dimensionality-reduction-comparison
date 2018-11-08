import numpy as np
from time import time
from sklearn import manifold

def fit_transform(X_highdimensional, n_neighbors=12):
    '''
        Performs Laplacian Eigenmaps on the given highdimensional dataset. The data needs to be a numpy array of the form (n_samples, n_features).

        X_highdimensional:  (n_samples, n_features) matrix containing the highdimensional datapoints.
        n_neighbors: The number of neighbours to consider for each point. Default: 12

        Returns a tuple (X_low, time) with the lowdimensional representation and the time the execution took.
    '''    
    laplacian_eigenmaps = manifold.SpectralEmbedding(n_components=2, affinity ="nearest_neighbors", 
                                                     random_state=None, n_neighbors=n_neighbors)
    t0 = time()
    X_low = laplacian_eigenmaps.fit_transform(X_highdimensional)
    return X_low, (time() - t0)