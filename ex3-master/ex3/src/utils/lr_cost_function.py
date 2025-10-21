import numpy as np
from .sigmoid import sigmoid
from numpy import size, zeros, log, sum



def lr_cost_function(theta, X, y, _lambda, alpha=1):
    """
    Logistic Regression Cost Function.

    Compute cost and gradient for logistic regression with regularization

    m = size(y)
    cost = 1/m * (sum(-y * log(h_x) - (1-y) * log(1-h_x))) + lambda / (2*m) * sum(theta[1:]^2)

    k = size(theta)
    regularized_gradient = [grad1, grad2, ... grad_k]

    :param theta: theta parameters of the model (shape: (n+1, 1) or (n+1,))
    :param X: training set (shape: (m, n+1))
    :param y: training set labels (shape: (m, 1) or (m,))
    :param _lambda: lambda for regularization
    :param alpha: alpha parameter for gradient (often learning rate, but not used in *cost* function itself)

    :return: (cost, gradient) for the given parameters of the model (gradient shape matches theta)
    """

    m = np.size(y) # Number of training examples
    y_pred = sigmoid(X @ theta)
    J = sum(-y * log(y_pred) - (1 - y) * log(1 - y_pred)) / m
    reg = np.sum(theta[1:] ** 2) * (_lambda / (2 * m))
    J += reg

    grad = (X.T @ (y_pred - y)) / m
    grad[1:] += (_lambda / m) * theta[1:]
    return J, grad
