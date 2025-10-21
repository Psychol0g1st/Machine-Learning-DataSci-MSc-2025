import scipy.io
from os import getcwd, path


def load_data():
    """
    Loads the training data to variables X, y.

    The training data can be found under /src/data/ex3data1.mat

    :return: X, y the training set and associated labels
    """

    print('Loading and Visualizing Data ...')

    file_name = path.join(getcwd(), 'ex3', 'src', 'data', 'ex3data1')
    data = scipy.io.loadmat(file_name)

    # training data stored in arrays X, y
    # y should be a row vector of labels
    return data['X'], data['y'].T[0]
