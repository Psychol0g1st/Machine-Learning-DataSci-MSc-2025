import scipy.io
from os import getcwd, path


def load_weights():
    """
    Loads the trained weights to (theta1, theta2)

    The trained weights can be found under /src/data/ex3weights.mat

    :return: (theta1, theta2) trained weights
    """

    print('Loading Saved Neural Network Parameters ...')

    file_name = path.join(getcwd(), 'ex3', 'src', 'data', 'ex3weights')
    weights = scipy.io.loadmat(file_name)

    return weights['Theta1'], weights['Theta2']
