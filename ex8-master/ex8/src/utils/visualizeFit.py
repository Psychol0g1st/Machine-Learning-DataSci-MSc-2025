import numpy as np
from matplotlib import pyplot
from .multivariateGaussian import multivariateGaussian


def visualizeFit(X, mu, sigma2):
    """
       visualizeFit Visualize the dataset and its estimated distribution.
       This visualization shows you the 
       probability density function of the Gaussian distribution. Each example
       has a location (x1, x2) that depends on its feature values.
    """
    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
    Z = multivariateGaussian(np.stack([X1.ravel(), X2.ravel()], axis=1), mu, sigma2)
    Z = Z.reshape(X1.shape)

    pyplot.plot(X[:, 0], X[:, 1], 'bx', mec='b', mew=2, ms=8)

    if np.all(abs(Z) != np.inf):
        pyplot.contour(X1, X2, Z, levels=10**(np.arange(-20., 1, 3)), zorder=100)
