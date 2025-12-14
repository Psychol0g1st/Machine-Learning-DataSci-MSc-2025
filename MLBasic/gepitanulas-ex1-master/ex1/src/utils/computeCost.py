import numpy as np


def computeCost(X, y, theta):
    # COMPUTECOST Compute cost for linear regression
    #   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    #   parameter for linear regression to fit the data points in X and y

    # Initialize some useful values
    m = y.shape[0]  # number of training examples

    # You need to return the following variables correctly
    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.

    # for i in range(m):
    #     y_pred = (theta[0]*X[i,0]+theta[1]*X[i,1])
    #     J+= (y_pred - y[i])**2
    # J = J/(2*m)
    J = np.sum((X.dot(theta)-y)**2)/(2*m)

    # =========================================================================

    return J
