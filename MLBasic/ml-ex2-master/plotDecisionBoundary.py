from numpy import size, min, max, array, linspace, zeros
import matplotlib.pyplot as plt
from plotData import plotData
from mapFeature import mapFeature

# PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
# the decision boundary defined by theta
#   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with * for the
#   positive examples and o for the negative examples. X is assumed to be
#   a either
#   1) Mx3 matrix, where the first column is an all-ones column for the
#      intercept.
#   2) MxN, N>3 matrix, where the first column is all-ones


def plotDecisionBoundary(theta, X, y):

    # Plot Data
    plotData(X[:, 1:3], y)

    if size(X, 1) <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = array([min(X[:, 1]) - 2, max(X[:, 1]) + 2])

        # Calculate the decision boundary line
        plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)

        # Legend, specific for the exercise
        plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        plt.xlim([30, 100])
        plt.ylim([30, 100])
    else:
        # Here is the grid range
        # u = linspace(-1, 1.5, 50)
        # v = linspace(-1, 1.5, 50)
        u = linspace(30, 100, 50)
        v = linspace(30, 100, 50)

        z = zeros((u.size, v.size))
        # Evaluate z = theta*x over the grid

        for i in range(size(u, 0)):
            for j in range(size(v, 0)):
                z[i, j] = mapFeature(array([u[i]]), array([v[j]]),3) @ theta


        z = z.T  # important to transpose z before calling contour

        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        plt.contour(u, v, z, levels=[0], linewidths=2, colors='c')