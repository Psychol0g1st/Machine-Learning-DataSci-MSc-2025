"""
Machine Learning Online Class - Exercise 1: Linear Regression

Instructions
------------
This file contains code that helps you get started on the
linear exercise. You will need to complete the following functions
in this exercise:
    warmUpExercise.m
    plotData.m
    gradientDescent.m
    computeCost.m
    gradientDescentMulti.m
    computeCostMulti.m
    featureNormalize.m
    normalEqn.m

For this exercise, you will not need to change any code in this file,
or any other files other than those mentioned above.

x refers to the population size in 10,000s
y refers to the profit in $10,000s

"""
from matplotlib.colors import LogNorm

from src.util import continue_or_quit
from src.utils.compute_cost import compute_cost
from src.utils.gradient_descent import gradient_descent
from src.utils.plot_data import plot_data
from src.utils.warm_up_exercise import warm_up_exercise
import numpy as np
import matplotlib.pyplot as plt


# ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.m

print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')
warm_up_exercise()

print('Program paused. Press enter to continue.\n')
continue_or_quit()

# ======================= Part 2: Plotting =======================

print('Plotting Data...')
data = np.loadtxt('./data/ex1data1.txt', delimiter=',', usecols=(0, 1))
X = data[:, 0]
y = data[:, 1]
m = y.size  # number of training data

# Plot Data
# Note: You have to complete the code in utils/plot_data.py
plot_data(X, y)

continue_or_quit()

# =================== Part 3: Cost and Gradient descent ===================

X = np.c_[np.ones(m), X]  # add a column of ones to X
theta = np.zeros(2)  # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function...\n')
# compute and display initial cost
J = compute_cost(X, y, theta)
print(f'With theta = [0 ; 0]\nCost computed = {J}\n')
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
J = compute_cost(X, y, np.array([-1, 2]))
print(f'\nWith theta = [-1 ; 2]\nCost computed = {J}\n')
print('Expected cost value (approx) 54.24\n')

continue_or_quit()

print('\nRunning Gradient Descent...\n')
# run gradient descent
theta, _ = gradient_descent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:\n')
print(f'{theta}\n')
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

# Plot the linear fit
plt.figure(0)
line1, = plt.plot(X[:, 1], np.dot(X, theta), label='Linear Regression')
plt.legend(handles=[line1])

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot(np.array([1, 3.5]), theta)
print('For population = 35,000, we predict a profit of {:0.3f}'.format(predict1*10000))
predict2 = np.dot(np.array([1, 7]), theta)
print('For population = 70,000, we predict a profit of {:0.3f}'.format(predict2*10000))

continue_or_quit()

# ============= Part 4: Visualizing J(theta_0, theta_1) =============

print('Visualizing J(theta0, theta1) ...')
# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
xs, ys = np.meshgrid(theta0_vals, theta1_vals)
J_vals = np.zeros(xs.shape)

# Fill out J_vals
for i in range(theta0_vals.size):
    for j in range(theta1_vals.size):
        t = np.array([theta0_vals[i], theta1_vals[j]])
        J_vals[i][j] = compute_cost(X, y, t)

# we need to transpose J_vals, otherwise the axis will be flipped
J_vals = np.transpose(J_vals)

# Surface plot
fig1 = plt.figure(1)
ax = fig1.gca(projection='3d')
ax.plot_surface(xs, ys, J_vals)
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')

# Contour plot
plt.figure(2)
plt.contour(xs, ys, J_vals, levels=np.logspace(-2, 3, 20), norm=LogNorm())
plt.plot(theta[0], theta[1], c='r', marker="x")

input('Finished.\nPress anything to exit.')
