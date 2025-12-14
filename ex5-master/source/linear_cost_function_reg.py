import numpy as np

def linear_cost_function_reg(theta, X, y, lambda_):
    """
    LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
    regression with multiple variables
      [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
      cost of using theta as the parameter for linear regression to fit the
      data points in X and y. Returns the cost in J and the gradient in grad

    """
    m = X.shape[0]
    cost=0
    gradient = np.zeros(theta.shape)
    """====================== YOUR CODE HERE ======================
    Instructions: Compute the cost and gradient of regularized linear 
                  regression for a particular choice of theta.
    
                  You should set J to the cost and grad to the gradient.
    
    """
      
    hyp = X.dot(theta)
    cost = np.sum((hyp-y)**2) * (1/(2*m))
    reg = np.sum(theta[1:]**2)/(2*m)
    cost += lambda_*reg

    gradient = (hyp-y).dot(X)/m
    gradient[1:] += lambda_*(theta[1:]/m)

    return cost, gradient
