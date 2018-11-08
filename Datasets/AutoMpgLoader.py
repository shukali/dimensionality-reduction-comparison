import pandas as pd
from sklearn import datasets

def load_autompg():
    ''' Gets the Auto MPG dataset. Returns a tuple (data, target) containing the dataset and the labels.

        data:   The 392 x 8 data matrix containing the features. (n_samples = 392, n_features = 8)
                Order: (mpg, cylinders, displacement, horsepower, weight, acceleration, model year, origin)
        target: The 392 x 1 label vector containing the names of the cars, each being unique

        https://archive.ics.uci.edu/ml/datasets/auto+mpg
        
        NOTE: The horsepower feature has 6 missing values which are marked with '?' in the original dataset. 
        The original dataset has 398 entries but we only use the complete entries (392).
    '''
    data = pd.read_csv('Datasets/AutoMPG/auto-mpg.csv', index_col='name')
    data = data[data.horsepower != '?']
    data.horsepower = data.horsepower.astype('float')
    return data.values, data.index