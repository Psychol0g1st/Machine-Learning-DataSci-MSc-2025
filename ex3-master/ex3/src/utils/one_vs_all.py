import numpy as np
import scipy.optimize as optimize

from .lr_cost_function import lr_cost_function


def one_vs_all(X, y, num_labels, lam):
    """
    One vs. All trains multiple logistic regression classifiers.

    Returns all the classifiers in a matrix all_theta, where the i-th row of all_theta corresponds to the classifier
    for label i.

    :param X: the training set
    :param y: the training set labels
    :param num_labels: number of possible classes
    :param lam: lambda for regularization

    :return: all the classifiers in a matrix all_theta, where the i-th row of all_theta corresponds to the classifier
             for label i
    """

    m = np.size(X, 0)  # 5000
    n = np.size(X, 1)  # 400

    """
    ============================================== YOUR CODE HERE ======================================================
    Instructions: You should complete the following code to train `num_labels` logistic regression classifiers
                  with regularization parameter lambda.
    
    Hint: to obtain the i_th row of theta --> theta[i, :]
    Hint: you can use (y == i) to create a True/False array of y, where y == i
    
    Note: for this assignment it is recommended to use `scipy.optimize.fmin_cg` to optimize the logistic regression
          cost function. For this purpose there are two helper function defined below (`cost` and `grad`).
          It is also okay to use a for loop to loop over different classes.
    
          Example usage of scipy.optimize.fmin_cg:
          ```
          import scipy.optimize as optimize
    
          for i in range(0, num_labels):
              trained_theta_i = optimize.fmin_cg(f=cost, fprime=grad, x0=theta_i, args=args, maxiter=50)
          ```
    """

    all_theta = np.zeros((num_labels, n + 1))  # 10 x 401
    X = np.c_[np.ones(m), X]  # Add ones to the X data matrix
    for i in range(num_labels):
        y_subset = np.zeros(m)
        y_subset[y == (i+1)] = 1
        all_theta[i, :] = optimize.fmin_cg(f=cost, fprime=grad, x0=all_theta[i, :], args=(X, y_subset, lam), maxiter=50)
    

    return all_theta


def cost(theta, *args):
    """
    Logistic Regression Cost function, used for fmin_cg during the one_vs_all training.

    :param theta: theta parameters of the model
    :param args: (x, y, lam) the training set, labels and lambda for regularization
    :return: cost with given parameters
    """
    x, y, lam = args
    (J, g) = lr_cost_function(theta, x, y, lam)
    return J


def grad(theta, *args):
    """
    Gradients based on the Logistic Regression Cost function, used for fmin_cg during the one_vs_all training.

    :param theta: theta parameters of the model
    :param args: (x, y, lam) the training set, labels and lambda for regularization
    :return: cost with given parameters
    """
    x, y, lam = args
    (J, g) = lr_cost_function(theta, x, y, lam)
    return g
