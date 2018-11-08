import numpy as np
from time import time
from sklearn import manifold

def fit_transform(X_highdimensional, n_different_init=4,  max_iteration=200, is_metric=True):
    '''
        Performs (Non-) Metric MDS on the given highdimensional dataset. The data needs to be a numpy array of the form (n_samples, n_features).

        X_highdimensional:  (n_samples, n_features) matrix containing the highdimensional datapoints.
        n_different_init: The number of times, the algorithm will be started with different initialization. The best result 
                          of the runs (the one with the lowest STRESS) will be taken. Default: 4
        max_iteration: Maximal number of iterations the algorithm will take in a single run. Default: 200
        is_metric: The boolean value determining whether to use Metric MDS or Non-Metric MDS.

        Returns a tuple (X_low, time, STRESS) with the lowdimensional representation, the time the execution took and the STRESS value.
    '''
    mds = manifold.MDS(n_components=2, metric=is_metric, n_init=n_different_init, max_iter=max_iteration, random_state=None)
    t0 = time()
    X_low = mds.fit_transform(X_highdimensional)
    return X_low, (time() - t0), mds.stress_