#!/usr/bin/env python3
from numpy import zeros, size, mean, loadtxt, ones
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from plotData import plotData
from mapFeature import mapFeature
from predict import predict
from costFunctionReg import costFunctionReg
from plotDecisionBoundary import plotDecisionBoundary, plotDecisionBoundaryScaled

# Machine Learning Online Class - Exercise 2: Logistic Regression
#
# Instructions
# ------------
#
# This file contains code that helps you get started on the second part
# of the exercise which covers regularization with logistic regression.
#
# You will need to complete the following functions in this exericse:
#
#     sigmoid.py
#     costFunction.py
#     predict.py
#     costFunctionReg.py
#
# For this exercise, you will not need to change any code in this file,
# or any other files other than those mentioned above.
#


def ex2_reg():

    # Initialization
    os.system("cls" if os.name == "nt" else "clear")

    # Load Data
    data = loadtxt("ex2data1.txt", delimiter=",")
    #  The first two columns contains the exam scores and the third column
    #  contains the label.
    X = data[:, 0:2]
    y = data[:, 2]

    plotData(X, y)

    # Labels and Legend
    # plt.xlabel("Microchip Test 1")
    # plt.ylabel("Microchip Test 2")

    # # Specified in plot order
    # plt.plot([], [], "bo", label="y = 1")
    # plt.plot([], [], "r*", label="y = 0")
    # plt.legend()
    # plt.show()

    # =========== Part 1: Regularized Logistic Regression ============
    #  In this part, you are given a dataset with data points that are not
    #  linearly separable. However, you would still like to use logistic
    #  regression to classify the data points.
    #
    #  To do so, you introduce more features to use -- in particular, you add
    #  polynomial features to our data matrix (similar to polynomial
    #  regression).
    #

    # Add Polynomial Features

    # Note that mapFeature also adds a column of ones for us, so the intercept
    # term is handled
    x_mean = mean(X, axis=0)
    x_std = X.std(axis=0)
    X = (X - x_mean) / x_std
    X = mapFeature(X[:, 0], X[:, 1])

    # Initialize fitting parameters
    initial_theta = zeros(X.shape[1], dtype=float)

    # Set regularization parameter lambda to 1
    _lambda = 1

    # Compute and display initial cost and gradient for regularized logistic
    # regression
    cost, grad = costFunctionReg(initial_theta, X, y, _lambda)
    print("Cost at initial theta (zeros): {:0.3f}".format(cost))
    print("Expected cost (approx): 0.693")
    print("Gradient at initial theta (zeros) - first five values only:")
    for g in grad[:5]:
        print("{:0.4f}".format(g))
    print("Expected gradients (approx) - first five values only:")
    print("0.0085\n0.0188\n0.0001\n0.0503\n0.0115")

    input("Program paused. Press enter to continue.")

    # Compute and display cost and gradient
    # with all-ones theta and lambda = 10
    test_theta = ones(X.shape[1])
    cost, grad = costFunctionReg(test_theta, X, y, 1)

    print("Cost at test theta (with lambda = 10): {:0.2f}".format(cost))
    print("Expected cost (approx): 3.16\n")
    print("Gradient at test theta - first five values only:")
    for g in grad[:5]:
        print("{:0.4f}".format(g))
    print("Expected gradients (approx) - first five values only:")
    print("0.3460\n0.1614\n0.1948\n0.2269\n0.0922")

    input("Program paused. Press enter to continue.")

    # ============= Part 2: Regularization and Accuracies =============
    #  Optional Exercise:
    #  In this part, you will get to try different values of lambda and
    #  see how regularization affects the decision coundart
    #
    #  Try the following values of lambda (0, 1, 10, 100).
    #
    #  How does the decision boundary change when you vary lambda? How does
    #  the training set accuracy vary?
    #

    # Initialize fitting parameters
    initial_theta = zeros(X.shape[1], dtype=float)

    # Set regularization parameter lambda to 1 (you should vary this)
    _lambda = 1

    # Set Options
    options = {"maxiter": 400}

    # Optimize
    result = minimize(fun=costFunctionReg,
                      x0=initial_theta,
                      args=(X, y, _lambda),
                      jac=True,
                      method="TNC",
                      options=options)

    theta = result.x

    # Plot Boundary
    plotDecisionBoundaryScaled(theta, X, y, x_mean, x_std)

    # Title
    plt.title("lambda = {:g}".format(_lambda))

    # Labels and Legend
    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")

    plt.plot([], [], "bo", label="y = 1")
    plt.plot([], [], "r*", label="y = 0")
    plt.plot([], [], "c-", label="Decision boundary")
    plt.legend()
    plt.show()

    # Compute accuracy on our training set
    p = predict(theta, X)
    print("Train Accuracy: {:6f}%".format(mean(p == y) * 100))

#############################################################################


if __name__ == "__main__":
    ex2_reg()
