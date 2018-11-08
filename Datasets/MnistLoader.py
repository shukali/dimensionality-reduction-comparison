import struct
import numpy as np
from sklearn.utils import shuffle

def load_mnist(n_samples=1000):
    ''' Gets the MNIST dataset. Returns a tuple (data, target) containing the dataset and the labels.

        data:   The 1000 x 784 data matrix containing the images. (n_samples = 1000, n_features =784)
        target: The 1000 x 1 label vector containing the labels for the images

        http://yann.lecun.com/exdb/mnist/
        
        NOTE: The dataset consists of 60.000 training images and 10.000 test images.
    '''
    # 1) Download at least the two training .gz from http://yann.lecun.com/exdb/mnist/
    # 2) Don't rename them
    # 3) Unpack them to the path 'Datasets/MNIST/'
    X_train_all = read_idx('Datasets/MNIST/train-images.idx3-ubyte') # load training images
    X_train_all = np.reshape(X_train_all, (60000, 784))
    Y_train_all = read_idx('Datasets/MNIST/train-labels.idx1-ubyte') # load training labels
    X_train, Y_train = shuffle(X_train_all, Y_train_all, n_samples=n_samples, random_state=1)
    return X_train, Y_train

def read_idx(filename):
    """ A function that can read MNIST's idx file format into numpy arrays.
        Credits to https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
    """
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)