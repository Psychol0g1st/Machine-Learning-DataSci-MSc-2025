# -*- coding: utf-8 -*-

import numpy as np

from .sigmoid import sigmoid

def predict(Theta1, Theta2, X):
    """
    outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """

# Useful values
    m = X.shape[0]
    p = 0
# You need to return the following variables correctly 
    h1 = sigmoid(np.c_[np.ones((m, 1)), X].dot(Theta1.T))
    h2 = sigmoid(np.c_[np.ones((m, 1)), h1].dot(Theta2.T))
    p = np.matrix(np.argmax(h2, axis = 1)).T
# =========================================================================

    return p + 1