# COMPUTECOST Compute cost for linear regression
#   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
#   parameter for linear regression to fit the data points in X and y
import numpy as np

def compute_cost(X, y, theta):
    # Initialize some useful values
    m = y.size

    # You need to return the following variable correctly
    J = 0

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta.
    #                You should set J to the cost.
    predictions = X.dot(theta)
    sqrErrors = (predictions - y) ** 2
    J = 1 / (2 * m) * np.sum(sqrErrors)
    # ==========================================================

    return J
