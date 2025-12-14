import numpy as np
from .sigmoid import sigmoid


def predict_nn(theta1, theta2, X):
    """
    Predict the label of an input given a trained neural network.

    :param theta1: trained theta1 for layer1
    :param theta2: trained theta2 for layer2
    :param X: list of inputs
    :return: the predicted label of X given the trained weights of a neural network (theta1, theta2)
    """

    m = np.size(X, 0)

    """
    ================================================ YOUR CODE HERE ====================================================
    Instructions: Complete the following code to make predictions using your learned neural network. You should set p
                  to a vector containing labels between 1 to num_labels.
                  
    Hint: you can use `np.argmax` to get the index of a given element in an array.
          Example usage when dealing with matrices
          ```
          p = np.argmax(output, axis=1)  #  returns a vector with the index of each column's maximum element
          ```
    """
    
    # 1. Input Layer (a1) -> Hidden Layer
    # Add bias unit (column of ones) to X. Shape becomes (m, n+1)
    a1 = np.c_[np.ones(m), X]
    
    # Calculate z2 and activation a2
    z2 = a1.dot(theta1.T)
    a2 = sigmoid(z2)

    # 2. Hidden Layer (a2) -> Output Layer
    # Add bias unit to the hidden layer activations. Shape becomes (m, hidden_units+1)
    a2 = np.c_[np.ones(m), a2]
    
    # Calculate z3 and final hypothesis h_theta
    z3 = a2.dot(theta2.T)
    h_theta = sigmoid(z3)

    # 3. Prediction
    # np.argmax returns the index (0, 1, 2...) of the max value.
    # Since your labels are likely 1-indexed (1 to 10), we add +1 to the result.
    p = np.argmax(h_theta, axis=1) + 1

    return p
