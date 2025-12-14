import os
import numpy as np
from matplotlib import pyplot
from scipy.io import loadmat
from os.path import join

from .utils import estimateGaussian, multivariateGaussian, visualizeFit,selectThreshold

def ex8_1():
    """
    Machine Learning Class - Exercise 8 | Part 1: Anomaly Detection 
    Instructions
    ------------

    This file contains code that helps you get started on the
    exercise. You will need to complete the following functions:

        estimateGaussian.py
        selectThreshold.py
        cofiCostFunc.py

    For this exercise, you will not need to change any code in this file,
    or any other files other than those mentioned above.
    """

    """
    ================== Part 1: Load Example Dataset  ===================
    We start this exercise by using a small dataset that is easy to
    visualize.

    Our example case consists of 2 network server statistics across
    several machines: the latency and throughput of each machine.
    This exercise will help us find possibly faulty (or very fast) machines.
    """


    print()
    print('--------------------------------')
    print('Anomaly Detection')
    print('--------------------------------')
    print()

    #  The following command loads the dataset.
    data = loadmat(os.path.join(os.getcwd(), 'ex8/src/data/ex8data1.mat'))
    X, Xval, yval = data['X'], data['Xval'], data['yval'][:, 0]

    #  Visualize the example dataset
    pyplot.plot(X[:, 0], X[:, 1], 'bx', mew=2, mec='k', ms=6)
    # pyplot.plot(Xval[yval==0, 0], Xval[yval==0, 1], 'gx')
    # pyplot.plot(Xval[yval == 1, 0], Xval[yval == 1, 1], 'rx' )
    pyplot.axis([0, 30, 0, 30])
    pyplot.xlabel('Latency (ms)')
    pyplot.ylabel('Throughput (mb/s)')
    pyplot.show()
    """
    ================== Part 2: Estimate the dataset statistics ===================
     For this exercise, we assume a Gaussian distribution for the dataset.

     We first estimate the parameters of our assumed Gaussian distribution, 
     then compute the probabilities for each of the points and then visualize 
     both the overall distribution and where each of the points falls in 
     terms of that distribution.
    """
    input('\nPress ENTER to start Estimate the dataset statistics')
    print('Estimate the dataset statistics...')

    #  Estimate mu and sigma2
    mu, sigma2 = estimateGaussian(X)

    #  Returns the density of the multivariate normal at each data point (row) 
    #  of X
    p = multivariateGaussian(X, mu, sigma2)

    #  Visualize the fit
    visualizeFit(X,  mu, sigma2)
    pyplot.xlabel('Latency (ms)')
    pyplot.ylabel('Throughput (mb/s)')
    pyplot.tight_layout()
    pyplot.show()

    """
    ================== Part 3: Find Outliers ===================
     Now you will find a good epsilon threshold using a cross-validation set
     probabilities given the estimated Gaussian distribution
    """
    input('\nPress ENTER to start Find Outliers')
    print('Find Outliers...')
    
    pval = multivariateGaussian(Xval, mu, sigma2)
    epsilon, F1 = selectThreshold(yval, pval)
    print('Best epsilon found using cross-validation: %.2e' % epsilon)
    print('Best F1 on Cross Validation Set:  %f' % F1)
    print('   (you should see a value epsilon of about 8.99e-05)')
    print('   (you should see a Best F1 value of  0.875000)')

    #  Find the outliers in the training set and plot the
    outliers = p < epsilon

    #  Visualize the fit
    visualizeFit(X,  mu, sigma2)
    pyplot.xlabel('Latency (ms)')
    pyplot.ylabel('Throughput (mb/s)')
    pyplot.tight_layout()

    #  Draw a red circle around those outliers
    pyplot.plot(X[outliers, 0], X[outliers, 1], 'ro', ms=10, mfc='None', mew=2)
    pyplot.show()
    """
     ================== Part 4: Multidimensional Outliers ===================
      We will now use the code from the previous part and apply it to a 
      harder problem in which more features describe each datapoint and only 
      some features indicate whether a point is an outlier.
    """
    input('\nPress ENTER to start Multidimensional Outliers')
    print('Multidimensional Outliers...')
    
    #  Loads the second dataset. You should now have the
    #  variables X, Xval, yval in your environment
    data = loadmat(os.path.join(os.getcwd(), 'ex8/src/data/ex8data2.mat'))

    X, Xval, yval = data['X'], data['Xval'], data['yval'][:, 0]

    # Apply the same steps to the larger dataset
    mu, sigma2 = estimateGaussian(X)

    #  Training set 
    p = multivariateGaussian(X, mu, sigma2)

    #  Cross-validation set
    pval = multivariateGaussian(Xval, mu, sigma2)

    #  Find the best threshold
    epsilon, F1 = selectThreshold(yval, pval)

    print('Best epsilon found using cross-validation: %.2e' % epsilon)
    print('Best F1 on Cross Validation Set          : %f\n' % F1)
    print('\n# Outliers found: %d' % np.sum(p < epsilon))
    print('  (you should see a value epsilon of about 1.38e-18)')
    print('   (you should see a Best F1 value of      0.615385)')

    