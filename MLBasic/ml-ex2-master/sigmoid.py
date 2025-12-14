#!/usr/bin/env python3
from numpy import zeros, size, exp, shape

# SIGMOID Compute sigmoid function
# sigmoid(z): computes the sigmoid of z.


def sigmoid(z):

    # You need to return the following variables correctly
    g = zeros(shape(z), dtype=float)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the sigmoid of each value of z (z can be a matrix,
    #             vector or scalar).
    g = 1/(1+exp(-z))

    return g
    # =============================================================
