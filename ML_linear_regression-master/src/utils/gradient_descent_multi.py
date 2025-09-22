import numpy as np


# GRADIENTDESCENTMULTI Performs gradient descent to learn theta
#   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
#   taking num_iters gradient steps with learning rate alpha
from utils.compute_cost import compute_cost


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.size
    J_history = np.zeros(num_iters)

    for i in range(0, num_iters):
        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCostMulti) and gradient here.
        predictions = X.dot(theta)
        errors = predictions - y
        theta = theta - (alpha / m) * (X.T.dot(errors))
        # ============================================================

        J_history[i] = compute_cost(X, y, theta)    # save the cost

    return theta, J_history
