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
    
    h = X.dot(theta)
    errors = h - y
    
    J_unreg = (1 / (2 * m)) * np.sum(np.square(errors))
    
    reg_cost_term = (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))
    
    cost = J_unreg + reg_cost_term
    
    grad_unreg = (1 / m) * (X.T.dot(errors))
    reg_grad_term = (lambda_ / m) * theta
    reg_grad_term[0] = 0  # Set the bias term's regularization to zero
    gradient = grad_unreg + reg_grad_term

    return cost, gradient
