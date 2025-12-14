import numpy as np
# COMPUTECOST Compute cost for linear regression
#   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
#   parameter for linear regression to fit the data points in X and y

def compute_cost(X, y, theta):
    # Initialize some useful values
    m = y.size

    # You need to return the following variable correctly
    J = 0

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta.
    #                You should set J to the cost.
    J = np.sum((X.dot(theta) - y) ** 2) / (2 * m)
    # ==========================================================

    return J
