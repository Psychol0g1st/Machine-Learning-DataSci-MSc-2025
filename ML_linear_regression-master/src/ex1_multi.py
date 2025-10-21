"""
 Exercise 1: Linear regression with multiple variables

 Instructions
 ------------

 This file contains code that helps you get started on the
 linear regression exercise.

 You will need to complete the following functions in this
 exercise:

    warmUpExercise.m
    plotData.m
    gradientDescent.m
    computeCost.m
    gradientDescentMulti.m
    computeCostMulti.m
    featureNormalize.m
    normalEqn.m

 For this part of the exercise, you will need to change some
 parts of the code below for various experiments (e.g., changing
 learning rates).

"""
import numpy as np
import matplotlib.pyplot as plt
from util import continue_or_quit
from utils.feature_normalize import feature_normalize
from utils.gradient_descent_multi import gradient_descent_multi
from utils.normal_eqn import normal_eqn


# ================ Part 1: Feature Normalization ================

print('Loading data ...\n')

# Load data
data = np.loadtxt('./data/ex1data2.txt', delimiter=',', dtype=np.int64)
X = data[:, 0:2]
y = data[:, 2]
m = y.size

# Print out some data points
print('First 10 examples from the dataset: ')
for i in range(0, 10):
    print('x = {}, y = {}'.format(X[i], y[i]))

continue_or_quit()

# Scale features and set them to zero mean
print('Normalizing Features ...')

X, mu, sigma = feature_normalize(X)
print(f'Mean: {mu}\n')
print(f'Standard deviation: {sigma}\n')
print(f'First 10 examples from the dataset after normalization: \nX = {X[0:10, :]}\n')
# Add intercept term to X
X = np.c_[np.ones(m), X]

"""
================ Part 2: Gradient Descent ================

====================== YOUR CODE HERE ======================
Instructions: We have provided you with the following starter
              code that runs gradient descent with a particular
              learning rate (alpha). 

              Your task is to first make sure that your functions - 
              computeCost and gradientDescent already work with 
              this starter code and support multiple variables.

              After that, try running gradient descent with 
              different values of alpha and see which one gives
              you the best result.

              Finally, you should complete the code at the end
              to predict the price of a 1650 sq-ft, 3 br house.

Hint: At prediction, make sure you do the same feature normalization.
"""

print('Running gradient descent ...')
# Choose some alpha value
alpha = 0.001
num_iters = 5000

# Init theta and Run Gradient Descent
theta = np.zeros(3)
theta, J_history = gradient_descent_multi(X, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.figure()
plt.plot(np.arange(J_history.size), J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

# Display gradient descent's result
print('Theta computed from gradient descent:\n')
print(f'{theta}\n\n')

# Estimate the price of a 1650 sq-ft, 3 br house
# ===================== Your Code Here =====================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.
new_input = np.array([1650, 3])
new_input = (new_input - mu) / sigma
new_input = np.r_[1, new_input]  # add intercept term
price = new_input.dot(theta)  # You should change this


# ==========================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent) : {:0.3f}'.format(price))
continue_or_quit()

# ===================== Part 3: Normal Equations =====================

print('Solving with normal equations...\n')

# ===================== Your Code Here =====================
# Instructions : The following code computes the closed form
#                solution for linear regression using the normal
#                equations. You should complete the code in
#                normalEqn.py
#
#                After doing so, you should complete this code
#                to predict the price of a 1650 sq-ft, 3 br house.

# Load data
data = np.loadtxt('./data/ex1data2.txt', delimiter=',', dtype=np.int64)
X = data[:, 0:2]
y = data[:, 2]
m = y.size

# Add intercept term to X
X = np.c_[np.ones(m), X]
# Calculate the parameters from the normal equation
theta = normal_eqn(X, y)
# Display normal equation's result
print('Theta computed from the normal equations : \n')
print(f'{theta}\n\n')

# Estimate the price of a 1650 sq-ft, 3 br house
# ===================== Your Code Here =====================
price = np.array([1, 1650, 3]).dot(theta)  # You should change this


# ==========================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations) : {:0.3f}'.format(price))

input('Finished.\nPress anything to exit.')
