import numpy as np
from .sigmoid import sigmoid


def predict_nn(theta1, theta2, X):
    """
    Predict the label of an input given a trained neural network.

    :param theta1: trained theta1 for layer1
    :param theta2: trained theta2 for layer2
    :param X: list of inputs
    :return: the predicted label of X given the trained weights of a neural network (theta1, theta2)
    """

    m = np.size(X, 0)

    """
    ================================================ YOUR CODE HERE ====================================================
    Instructions: Complete the following code to make predictions using your learned neural network. You should set p
                  to a vector containing labels between 1 to num_labels.
                  
    Hint: you can use `np.argmax` to get the index of a given element in an array.
          Example usage when dealing with matrices
          ```
          p = np.argmax(output, axis=1)  #  returns a vector with the index of each column's maximum element
          ```
    """

    #  index 0 means it's in class 1
    p = np.zeros(m)
    return p
