import numpy as np
from time import time
from sklearn import manifold

def fit_transform(X_highdimensional, perplexity=30.0):
    '''
        Performs t-SNE on the given highdimensional dataset. The data needs to be a numpy array of the form (n_samples, n_features).

        X_highdimensional:  (n_samples, n_features) matrix containing the highdimensional datapoints.
        perplexity: Quoting from https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html:
                    "The perplexity is related to the number of nearest neighbors that is used in other manifold 
                    learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting 
                    a value between 5 and 50. The choice is not extremely critical since t-SNE is quite 
                    insensitive to this parameter." Default: 30.0

        Returns a tuple (X_low, time) with the lowdimensional representation and the time the execution took.
    '''
    tsne = manifold.TSNE(n_components=2, perplexity=perplexity, random_state=None)
    t0 = time()
    X_low = tsne.fit_transform(X_highdimensional)
    return X_low, (time() - t0)