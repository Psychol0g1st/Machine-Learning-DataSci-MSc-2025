from numpy import exp, log, zeros, size, sum
from sigmoid import sigmoid

# COSTFUNCTION Compute cost and gradient for logistic regression
#   COSTFUNCTION(theta, X, y) computes the cost of using theta as the
#   parameter for logistic regression and the gradient of the cost
#   w.r.t. to the parameters.

def costFunction(theta, X, y):

    # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = zeros(size(theta, 0))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Note: grad should have the same dimensions as theta
    #

    h = sigmoid(X @ theta)  # Hypothesis function
    
    # Compute cost function J
    J = (-1/m) * (y.dot(log(h)) + (1 - y).dot(log(1 - h)))
    
    # Compute gradients
    grad = (1/m) * X.T.dot(h - y)

    
    return J, grad
    # =============================================================
