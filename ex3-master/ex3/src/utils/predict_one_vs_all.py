import numpy as np


def predict_one_vs_all(all_theta, X):
    """
    Predict the label for a trained one-vs-all classifier.

    :param all_theta: matrix containing all thetas for each class. The i_th row contains the thetas for class i
    :param X: the training set

    :return: p vector of predictions for each X element
    """

    m = np.size(X, 0)
    num_labels = np.size(all_theta, 0)

    # Initialize prediction vector p
    p = np.zeros(m)

    # Add ones to the X data matrix (since all_theta includes the intercept/bias term)
    X = np.c_[np.ones(m), X]  # X is now m x (n+1)

    """
    ================================================ YOUR CODE HERE ====================================================
    Instructions: Complete the following code to make predictions using your learned logistic regression
                  parameters (one-vs-all). You should set p to a vector of predictions, p[i] in range(1, num_labels).

    Hint: you can use `np.argmax` to get the index of a given element in an array.
          Example usage when dealing with a vector
          ```
          p = np.argmax(output)  #  returns the index of the output vector's maximum element
          ```
    """

    # 1. Calculate the hypothesis (sigmoid output) for all classes simultaneously.
    # The hypothesis is: H = X * all_theta.T
    # X: (m x n+1)
    # all_theta.T: (n+1 x num_labels)
    # H: (m x num_labels) -> H[i, j] is the probability that example X[i] belongs to class j (or j+1, see below).

    # The sigmoid function is used to convert the linear output to a probability.
    # Since sigmoid(z) = 1 / (1 + exp(-z)), and all_theta is applied to the features,
    # the probability matrix is:
    # H = 1 / (1 + exp(-(X @ all_theta.T)))

    # For prediction, we only need the *linear* score (z = X @ all_theta.T)
    # because the sigmoid function is monotonically increasing.
    # argmax(sigmoid(z)) is equivalent to argmax(z).

    Z = X @ all_theta.T  # (m x num_labels) matrix of linear scores
    max_score_indices = np.argmax(Z, axis=1)
    p = max_score_indices + 1

    return p
