import numpy as np


# NORMALEQN Computes the closed-form solution to linear regression
#   NORMALEQN(X,y) computes the closed-form solution to linear 
#   regression using the normal equations.

def normal_eqn(X, y):
    theta = np.zeros((X.shape[1], 1))
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the code to compute the closed form solution
    #               to linear regression and put the result in theta.
    #
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
    return theta
