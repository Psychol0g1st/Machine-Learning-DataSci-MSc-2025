import numpy as np
from matplotlib import pyplot as plot

from .poly_features import poly_features
from .normalize_feature import normalize_features

def plot_fit(min_x, max_x, mu, sigma, theta, power):
    x = np.arange(min_x - 15, max_x + 25, 0.05)
    Xpoly = poly_features(x, power)
    Xpoly, _, _ = normalize_features(Xpoly, mu, sigma)
    plot.plot(x, Xpoly.dot(theta), '--')