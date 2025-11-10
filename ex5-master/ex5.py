from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plot
from scipy.io import loadmat

from source.linear_cost_function_reg import linear_cost_function_reg
from source.train_linear_regression import train_linear_regression
from source.learning_curve import learning_curve
from source.poly_features import poly_features
from source.normalize_feature import normalize_features
from source.plot_fit import plot_fit
from source.validation_curve import validation_curve

def ex5():
    """ Exercise 5 | Regularized Linear Regression and Bias-Variance

     Instructions
     ------------

     This file contains code that helps you get started on the
     exercise. You will need to complete the following functions:

        linearRegCostFunction.m
        learningCurve.m
        validationCurve.m

     For this exercise, you will not need to change any code in this file,
     or any other files other than those mentioned above.

    """

    '''%% =========== Part 1: Loading and Visualizing Data =============
    We start the exercise by first loading and visualizing the dataset. 
    The following code will load the dataset into your environment and plot
    the data.'''

    print('Loading and Visualizing Data ...')
    data = loadmat('data/ex5data1.mat')
    X = data['X']
    y = data['y'].flatten()
    Xval = data['Xval']
    yval = data['yval'].flatten()

    m = X.shape[0]
    plot.plot(X, y, 'rx', markersize=10)
    plot.xlabel('Change in water level (x)')
    plot.ylabel('Water flowing out of the dam (y)')
    plot.show()

    print('Program paused. Press enter to continue.\n')
    input()

    """% =========== Part 2: Regularized Linear Regression Cost =============
    You should now implement the cost function and the gradient 
    for regularized linear regression. """

    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    Xval = np.concatenate((np.ones((Xval.shape[0], 1)), Xval), axis=1)
    theta = np.array([1.0, 1.0])
    # linear regression cost and gradient
    cost, gradient = linear_cost_function_reg(theta, X, y, 1.0)
    print('Cost at theta = [1, 1]: %f' % cost)
    print('(this value should be about 303.993192)')

    print('Gradient at theta = [1, 1]: \n %s' % gradient)
    print('(this value should be about [-15.303016, 598.250744])')

    print('Program paused. Press enter to continue.\n')
    input()

    """% =========== Part 3: Train Linear Regression =============
     Once you have implemented the cost and gradient correctly, the
     trainLinearReg function will use your cost function to train 
     regularized linear regression.
    
     Write Up Note: The data is non-linear, so this will not give a great 
                    fit.
    
    """
    lambda_ = 0.0
    theta = train_linear_regression(X, y, lambda_)
    plot.plot(X[:, 1], y, 'rx', markersize=10)
    plot.plot(X[:, 1], X.dot(theta), '--')
    plot.xlabel('Change in water level (x)')
    plot.ylabel('Water flowing out of the dam (y)')
    plot.show()

    print('Program paused. Press enter to continue.\n')
    input()

    """% =========== Part 4: Learning Curve for Linear Regression =============
     Next, you should implement the learningCurve function. 
    
     Write Up Note: Since the model is underfitting the data, we expect to
                    see a graph with "high bias" -- Figure 3 in ex5-solution.pdf 
    """

    lambda_ = 0.0
    err_train, err_val = learning_curve(X, y, Xval, yval, lambda_)
    plot.plot(range(m + 1), err_train)
    plot.plot(range(m + 1), err_val)
    plot.title('Learning curve for linear regression')
    plot.xlabel('Number of training examples')
    plot.ylabel('Error')
    plot.show()

    print('# Training Examples\tTrain Error\tCross Validation Error')
    for i in range(m + 1):
        print('  \t%d\t\t%f\t%f' % (i, err_train[i], err_val[i]))

    print('Program paused. Press enter to continue.\n')
    input()

    """% =========== Part 5: Feature Mapping for Polynomial Regression =============
     One solution to this is to use polynomial regression. You should now
     complete polyFeatures to map each example into its powers
    
    """
    power = 8
    Xpoly = poly_features(X[:, 1], power)
    Xpoly, mu, sigma = normalize_features(Xpoly)
    Xval_poly = poly_features(Xval[:, 1], power)
    Xval_poly, _, _ = normalize_features(Xval_poly)
    print('Normalized Training Example 1:\n%s' % Xpoly[0, :])

    print('Program paused. Press enter to continue.\n')
    input()

    """% =========== Part 6: Learning Curve for Polynomial Regression =============
     Now, you will get to experiment with polynomial regression with multiple
     values of lambda. The code below runs polynomial regression with 
     lambda = 0. You should try running the code with different values of
     lambda to see how the fit and learning curve change.
    """

    lambda_ = 0.0
    theta = train_linear_regression(Xpoly, y, lambda_)
    plot.plot(X[:, 1], y, 'rx', markersize=10)
    plot_fit(X.min(), X.max(), mu, sigma, theta, power)
    plot.xlabel('Change in water level (x)')
    plot.ylabel('Water flowing out of the dam (y)')
    plot.title('Polynomial Regression Fit (lambda = %f)' % lambda_)
    plot.show()

    print('Program paused. Press enter to continue.\n')
    input()

    # learning curve for polynomial fit
    err_train, err_val = learning_curve(Xpoly, y, Xval_poly, yval, lambda_)
    plot.plot(range(m + 1), err_train)
    plot.plot(range(m + 1), err_val)
    plot.title('Polynomial Regression Learning Curve (lambda = %f)' % (lambda_))
    plot.xlabel('Number of training examples')
    plot.ylabel('Error')
    plot.show()
    print('# Training Examples\tTrain Error\tCross Validation Error')
    for i in range(m + 1):
        print('  \t%d\t\t%f\t%f' % (i, err_train[i], err_val[i]))

    print('Program paused. Press enter to continue.\n')
    input()

    """% =========== Part 7: Validation for Selecting Lambda =============
     You will now implement validationCurve to test various values of 
     lambda on a validation set. You will then use this to select the
     "best" lambda value.
    
    """
    lambda_vec, err_train, err_val = validation_curve(Xpoly, y, Xval_poly, yval)
    plot.plot(lambda_vec, err_train)
    plot.plot(lambda_vec, err_val)
    plot.xlabel('lambda')
    plot.ylabel('Error')
    plot.show()
    print('# lambda\tTrain Error\tCross Validation Error')
    for i, lambda_ in enumerate(lambda_vec):
        print('  \t%f\t\t%f\t%f' % (lambda_, err_train[i], err_val[i]))
