import numpy as np
import matplotlib.pyplot as plt
from .utils import computeCost, gradientDescent, warmUpExercise, plotData


# Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exercise:
#
#     warmUpExercise.m
#     plotData.m
#     gradientDescent.m
#     computeCost.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#

def ex1():
    # ==================== Part 1: Basic Function ====================
    # Complete warmUpExercise.m

    print('Running warmUpExercise ...')
    print('5x5 Identity Matrix: \n ', warmUpExercise())

    print('Program paused. Press enter to continue.\n')
    input()

    # ======================= Part 2: Plotting =======================
    print('Plotting Data ...\n')
    data = np.loadtxt("./ex1/src/data/ex1data1.txt")
    X = data[:, 0]
    y = data[:, 1]
    m = y.shape[0]  # number of training examples

    # Plot Data
    # Note: You have to complete the code in plotData.m
    plotData(X, y)

    print('Program paused. Press enter to continue.\n')
    input()

    # =================== Part 3: Cost and Gradient descent ===================

    X = np.c_[np.ones((m, 1)), X]  # Add a column of ones to x
    theta = np.zeros(2)  # initialize fitting parameters

    # Some gradient descent settings
    iterations = 1500
    alpha = 0.01

    print('\nTesting the cost function ...\n')
    # compute and display initial cost
    J = computeCost(X, y, theta)
    print('With theta = [0 ; 0]\nCost computed =', J)
    print('Expected cost value (approx) 32.07\n')

    # further testing of the cost function
    J = computeCost(X, y, [-1, 2])
    print('\nWith theta = [-1 ; 2]\nCost computed =', J)
    print('Expected cost value (approx) 54.24\n')

    print('Program paused. Press enter to continue.\n')
    input()

    print('\nRunning Gradient Descent ...\n')
    # run gradient descent
    theta = gradientDescent(X, y, theta, alpha, iterations)

    # print theta to screen
    print('Theta found by gradient descent:\n')
    print(theta)
    print('Expected theta values (approx)\n')
    print(' -3.6303\n  1.1664\n\n')

    # Plot the linear fit
    plt.plot(X[:, 1], X.dot(theta))
    plt.legend(['Training data','Linear regression'])
    plt.show()

    # Predict values for population sizes of 35,000 and 70,000
    predict1 = np.array([1, 3.5]).dot(theta)
    print('For population = 35,000, we predict a profit of', predict1 * 10000)
    predict2 = np.array([1, 7]).dot(theta)
    print('For population = 70,000, we predict a profit of', predict2 * 10000)

    print('Program paused. Press enter to continue.\n')
    input()

    # ============= Part 4: Visualizing J(theta_0, theta_1) =============
    print('Visualizing J(theta_0, theta_1) ...')

    # Grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    # initialize J_vals to a matrix of 0's
    J_vals = np.zeros([theta0_vals.size, theta1_vals.size])

    # Fill out J_vals
    for i in range(theta0_vals.size):
        for j in range(theta1_vals.size):
            t = [theta0_vals[i], theta1_vals[j]]
            J_vals[i, j] = computeCost(X, y, t)

    # Because of the way meshgrids work in the surf command, we need to
    # transpose J_vals before calling surf, or else the axes will be flipped
    J_vals = np.transpose(J_vals)

    # Surface plot
    X, Y = np.meshgrid(theta0_vals, theta1_vals)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, J_vals, cmap='viridis', edgecolor='none')
    ax.set_title('Surface plot')
    plt.show()

    # Contour plot
    # Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    plt.contour(X, Y, J_vals, np.logspace(-2, 3, 20))
    plt.plot(theta[0], theta[1], 'rx')
    plt.show()
