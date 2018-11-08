from sklearn import datasets

def load_swissroll(n_datapoints=1000,  noice=0.0):
    ''' Loads the Swiss roll dataset with 1000, zero-varianced datapoints. Returns a tuple (data, target) containing the dataset and the labels.

        data:   The 1000 x 3 data matrix containing the points.
        target: The univariate position of the sample according to the main dimension of the points in the manifold. Can be used as the color.

        http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_swiss_roll.html   
    '''
    return datasets.make_swiss_roll(n_samples=n_datapoints, noise=noice, random_state=None);