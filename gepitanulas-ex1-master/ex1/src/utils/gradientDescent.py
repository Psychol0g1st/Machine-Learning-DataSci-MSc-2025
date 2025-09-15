import numpy as np
from .computeCost import computeCost

def gradientDescent(X, y, theta, alpha, num_iters):
    # GRADIENTDESCENT Performs gradient descent to learn theta
    #   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
    #   taking num_iters gradient steps with learning rate alpha

    # Initialize some useful values
    m = y.shape[0] # number of training examples
    J_history = np.zeros(num_iters)

    for iter in range(num_iters):
        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #
        # tmp = [0 0];
        # tmp(1) = theta(1) - alpha*(sum((X*theta-y)*X(:,1)')/m);
        # tmp(2) = theta(2) - alpha*(sum((X*theta-y)*X(:,2)')/m);
        # theta = tmp;



        # ============================================================

        # Save the cost J in every iteration
        J_history[iter] = computeCost(X, y, theta)

    return theta
