from sklearn import datasets

def load_iris():
    ''' Gets the Iris plants dataset. Returns a tuple (data, target) containing the dataset and the labels.

        data:   The 150 x 4 data matrix containing the single plants. (n_samples = 150, n_features = 4)
        target: The 150 x 1 label vector containing the classes for the plants (Iris-Setosa, Iris-Versicolour or Iris-Virginica)

        http://scikit-learn.org/stable/datasets/index.html#iris-plants-dataset     
    '''
    return datasets.load_iris(return_X_y = True)